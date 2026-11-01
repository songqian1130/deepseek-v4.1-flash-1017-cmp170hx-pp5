#!/usr/bin/env bash
# Launch an isolated five-GPU PP5 test container; never touches existing services.
set -Eeuo pipefail

MODE=${1:-baseline}
ROOT=/srv/models/deploy-llm/isolated-aiultrayang-pp5-ssd-async-20261031
IMAGE=local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-20261031
MODEL=/srv/models/DeepSeek-V4.1-Flash-1017-official
ALT=/home/llm/.cache/deepseek-v41-official
PORT=${PORT:-18020}
BASE=http://127.0.0.1:${PORT}
BASELINE_NAME=deepseek-v41-aiultrayang-pp5-ssd-nodspark-20261031
DS_NAME=deepseek-v41-aiultrayang-pp5-ssd-dspark-20261031

case "$MODE" in
  baseline)
    NAME=${CONTAINER_NAME:-$BASELINE_NAME}
    MAX_LEN=32768
    MAX_BATCHED=1025
    KV_BYTES=2147483648
    GPU_UTIL=0.95
    CPU_OFFLOAD=0,0,0,0,0
    SPEC=()
    ;;
  dspark)
    NAME=${CONTAINER_NAME:-$DS_NAME}
    MAX_LEN=8192
    MAX_BATCHED=1025
    KV_BYTES=536870912
    GPU_UTIL=0.95
    # Reserve 2 GiB of host-backed weight offload for the tight final PP rank.
    CPU_OFFLOAD=0,0,0,0,2
    SPEC=(--speculative-config '{"method":"dspark","num_speculative_tokens":3,"use_local_argmax_reduction":true}')
    ;;
  *) echo "Usage: $0 [baseline|dspark]" >&2; exit 2 ;;
esac

mkdir -p "$ROOT/reports" "$ROOT/cache"
LOG="$ROOT/reports/${MODE}-run.log"
exec > >(tee -a "$LOG") 2>&1
START=$(date +%s)
KEEP_RUNNING=0
heartbeat() {
  while sleep 20; do
    EL=$(( $(date +%s) - START ))
    STATUS=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo absent)
    LOG_PATH=$(docker inspect -f '{{.LogPath}}' "$NAME" 2>/dev/null || true)
    LOG_BYTES=$(stat -c %s "$LOG_PATH" 2>/dev/null || echo 0)
    echo "[HB ${EL}s] phase=monitor mode=$MODE container=$STATUS log_bytes=$LOG_BYTES"
  done
}
heartbeat & HB_PID=$!
cleanup_hb() { kill "$HB_PID" 2>/dev/null || true; wait "$HB_PID" 2>/dev/null || true; }
stop_own_container() { docker stop --time 120 "$NAME" >/dev/null 2>&1 || true; }
on_exit() {
  RC=$?
  cleanup_hb
  if (( RC != 0 && KEEP_RUNNING == 0 )); then
    echo "[HB] phase=cleanup stopping_only=$NAME exit=$RC"
    stop_own_container
  fi
}
trap on_exit EXIT
trap 'exit 130' INT TERM
fail() { echo "[HB] phase=RUN_FAIL reason=$*"; exit 1; }

if [[ "$MODE" == dspark ]] && [[ $(docker inspect -f '{{.State.Running}}' "$BASELINE_NAME" 2>/dev/null || echo false) == true ]]; then
  fail "isolated baseline is still using GPUs; stop only that isolated container before DSpark"
fi
echo "[HB 0s] phase=preflight mode=$MODE port=$PORT image=$IMAGE"
docker image inspect "$IMAGE" >/dev/null || fail "isolated image is not built"
if docker inspect "$NAME" >/dev/null 2>&1; then
  fail "container name already exists; refusing to replace it: $NAME"
fi
if ss -lnt | grep -q ":${PORT} "; then fail "port ${PORT} is already occupied"; fi
for path in "$MODEL/config.json" "$MODEL/model-00048-of-00048.safetensors" "$ALT/model-00048-of-00048.safetensors"; do
  [[ -r "$path" ]] || fail "checkpoint path unavailable: $path"
done
# Refuse to contend for any occupied GPU; never stop an existing process.
GPU_APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed '/^[[:space:]]*$/d')
[[ -z "$GPU_APPS" ]] || fail "GPU compute processes are already present; no existing process will be stopped"

ARGS=(
  serve "$MODEL"
  --host=127.0.0.1 --port="$PORT"
  --served-model-name=deepseek-v4.1-flash-1017
  --pipeline-parallel-size=5 --tensor-parallel-size=1
  --max-model-len="$MAX_LEN" --max-num-seqs=1 --max-num-batched-tokens="$MAX_BATCHED"
  --gpu-memory-utilization="$GPU_UTIL" --kv-cache-memory="$KV_BYTES"
  --kv-cache-dtype=fp8_ds_mla
  --trust-remote-code --disable-custom-all-reduce --enable-prefix-caching
  --safetensors-load-strategy=lazy
  --engram-config '{"cpu_offload":true,"storage":"ssd","engram_cache_gib":8}'
  --tokenizer-mode=deepseek_v41
  --enable-auto-tool-choice --tool-call-parser=deepseek_v41 --reasoning-parser=deepseek_v41
  --enable-prompt-tokens-details
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2],"max_cudagraph_capture_size":2}'
  "${SPEC[@]}"
)

echo "[HB] phase=container-start name=$NAME pp=5 layers=8,8,8,8,8 speculative=$MODE"
docker run -d --init --name "$NAME" \
  --gpus '"device=0,1,2,3,4"' --network host --shm-size=16g \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 --stop-timeout=600 \
  --memory=112g --memory-swap=112g \
  -v "$MODEL:$MODEL:ro" -v "$ALT:$ALT:ro" \
  -v "$ROOT/cache:$ROOT/cache:rw" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_PP_LAYER_PARTITION=8,8,8,8,8 \
  -e VLLM_CPU_OFFLOAD_GB_PER_RANK="$CPU_OFFLOAD" \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 \
  -e TMPDIR=/tmp -e PYTHONUNBUFFERED=1 \
  --entrypoint=/usr/local/bin/vllm "$IMAGE" "${ARGS[@]}"

LOG_PATH=$(docker inspect -f '{{.LogPath}}' "$NAME")
echo "[HB] phase=loading deadline_s=7200 log=$LOG_PATH"
DEADLINE=7200
LAST_BYTES=-1
LAST_CHANGE=$(date +%s)
while true; do
  NOW=$(date +%s)
  EL=$((NOW - START))
  STATE=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo missing)
  SIZE=$(stat -c %s "$LOG_PATH" 2>/dev/null || echo 0)
  if [[ "$SIZE" != "$LAST_BYTES" ]]; then LAST_BYTES=$SIZE; LAST_CHANGE=$NOW; fi
  if [[ "$STATE" != running ]]; then
    echo "[HB ${EL}s] phase=container-exited status=$STATE"
    docker inspect "$NAME" --format 'exit={{.State.ExitCode}} error={{.State.Error}} oom={{.State.OOMKilled}}' || true
    docker logs --tail 120 "$NAME" 2>&1 || true
    fail "isolated container exited during model load"
  fi
  if curl -fsS --max-time 3 "$BASE/health" >/dev/null 2>&1; then
    MODELS=$(curl -fsS --max-time 5 "$BASE/v1/models" 2>/dev/null || true)
    if [[ "$MODELS" == *deepseek-v4.1-flash-1017* ]]; then
      echo "[HB ${EL}s] phase=health-ready model=deepseek-v4.1-flash-1017"
      printf '%s\n' "$MODELS" > "$ROOT/reports/${MODE}-models.json"
      break
    fi
  fi
  if (( EL >= DEADLINE )); then
    echo "[HB ${EL}s] phase=deadline-stop-only container=$NAME"
    docker logs --tail 120 "$NAME" 2>&1 || true
    fail "120-minute model startup deadline reached"
  fi
  LAST=$(docker logs --tail 1 "$NAME" 2>&1 | tr '\r' '\n' | tail -1 | cut -c1-180)
  echo "[HB ${EL}s] phase=loading silence=$((NOW-LAST_CHANGE))s log_bytes=$SIZE last=$LAST"
  sleep 20
done

PHASE=smoke
timeout --foreground 1800 python3 "$ROOT/verify-pp5-ssd-async.py" \
  --base "$BASE/v1" --model deepseek-v4.1-flash-1017 \
  --out "$ROOT/reports/${MODE}-smoke.json"
KEEP_RUNNING=1
echo "[HB] phase=RUN_DONE mode=$MODE container=$NAME api=$BASE/v1"
