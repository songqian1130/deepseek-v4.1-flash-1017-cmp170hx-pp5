#!/usr/bin/env bash
# Isolated PP5 DSpark adaptation: keep the balanced 8,8,8,8,8 split and
# enable only a small, effective UVA budget on the last PP rank. This script
# never replaces or stops containers it did not create.
set -Eeuo pipefail

ROOT=${ROOT:-/srv/models/deploy-llm/isolated-aiultrayang-pp5-dspark-corrected-r4-20261031}
IMAGE=${IMAGE:-local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-dspark-head-uva-20261031}
MODEL=/srv/models/DeepSeek-V4.1-Flash-1017-official
ALT=/home/llm/.cache/deepseek-v41-official
NAME=${NAME:-deepseek-v41-aiultrayang-pp5-dspark-corrected-r4-20261031}
PORT=${PORT:-18028}
BASE=http://127.0.0.1:${PORT}
PARTITION=${PARTITION:-8,8,8,8,8}
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-5}
CPU_OFFLOAD=${CPU_OFFLOAD_PER_RANK:-0,0,0,0,5}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-}
REPORT_DIR=${ROOT}/reports
START=$(date +%s)
DEADLINE=${STARTUP_DEADLINE:-3600}
KEEP_RUNNING=0

mkdir -p "$REPORT_DIR" "$ROOT/cache"
exec > >(tee -a "$REPORT_DIR/dspark-r4-run.log") 2>&1

heartbeat() {
  while sleep 20; do
    EL=$(( $(date +%s) - START ))
    STATUS=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo absent)
  echo "[HB ${EL}s] phase=monitor mode=dspark-balanced container=$STATUS partition=$PARTITION cpu_offload_gb=$CPU_OFFLOAD_GB per_rank=$CPU_OFFLOAD p2p_disable=$NCCL_P2P_DISABLE p2p_level=$NCCL_P2P_LEVEL port=$PORT"
  done
}
heartbeat & HB_PID=$!
cleanup_heartbeat() { kill "$HB_PID" 2>/dev/null || true; wait "$HB_PID" 2>/dev/null || true; }
stop_own_container() { docker stop --time 120 "$NAME" >/dev/null 2>&1 || true; }
on_exit() {
  RC=$?
  cleanup_heartbeat
  if (( RC != 0 && KEEP_RUNNING == 0 )); then
    echo "[HB] phase=cleanup stopping_only=$NAME exit=$RC"
    stop_own_container
  fi
}
trap on_exit EXIT
trap 'exit 130' INT TERM
fail() { echo "[HB] phase=RUN_FAIL reason=$*"; exit 1; }

if docker inspect "$NAME" >/dev/null 2>&1; then
  fail "container name already exists; refusing to replace it: $NAME"
fi
if ss -lnt | grep -q ":${PORT} "; then
  fail "port ${PORT} is already occupied"
fi
if [[ $(docker inspect -f '{{.State.Running}}' deepseek-v41-aiultrayang-pp5-ssd-nodspark-20261031 2>/dev/null || echo false) == true ]]; then
  fail "isolated baseline is still using GPUs"
fi
if [[ $(docker inspect -f '{{.State.Running}}' deepseek-v41-aiultrayang-pp5-ssd-dspark-20261031 2>/dev/null || echo false) == true ]]; then
  fail "previous DSpark test is still using GPUs"
fi
if [[ $(docker inspect -f '{{.State.Running}}' deepseek-v41-aiultrayang-pp5-adapted-20261031 2>/dev/null || echo false) == true ]]; then
  fail "previous adapted test is still using GPUs"
fi
GPU_APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed '/^[[:space:]]*$/d')
[[ -z "$GPU_APPS" ]] || fail "GPU compute processes are already present; no existing process will be stopped"
docker image inspect "$IMAGE" >/dev/null || fail "isolated image is not built"
for path in "$MODEL/config.json" "$MODEL/model-00048-of-00048.safetensors" "$ALT/model-00048-of-00048.safetensors"; do
  [[ -r "$path" ]] || fail "checkpoint path unavailable: $path"
done

ARGS=(
  serve "$MODEL"
  --host=127.0.0.1 --port="$PORT"
  --served-model-name=deepseek-v4.1-flash-1017
  --pipeline-parallel-size=5 --tensor-parallel-size=1
  --max-model-len=8192 --max-num-seqs=1 --max-num-batched-tokens=1025
  --gpu-memory-utilization=0.95 --kv-cache-memory=536870912
  --kv-cache-dtype=fp8_ds_mla
  --cpu-offload-gb="$CPU_OFFLOAD_GB"
  --trust-remote-code --disable-custom-all-reduce --enable-prefix-caching
  --safetensors-load-strategy=lazy
  --engram-config '{"cpu_offload":true,"storage":"ssd","engram_cache_gib":8}'
  --tokenizer-mode=deepseek_v41
  --enable-auto-tool-choice --tool-call-parser=deepseek_v41 --reasoning-parser=deepseek_v41
  --enable-prompt-tokens-details
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2],"max_cudagraph_capture_size":2}'
  --speculative-config '{"method":"dspark","num_speculative_tokens":3,"use_local_argmax_reduction":true}'
)

echo "[HB 0s] phase=preflight image=$IMAGE name=$NAME partition=$PARTITION cpu_offload_gb=$CPU_OFFLOAD_GB cpu_offload_per_rank=$CPU_OFFLOAD port=$PORT"
echo "[HB] phase=rank-guard-confirmed dspark-drafter=last-pp-rank-only source=image-and-src-verified"
echo "[HB] phase=container-start name=$NAME pp=5 layers=$PARTITION speculative=dspark"
docker run -d --init --name "$NAME" \
  --gpus '"device=0,1,2,3,4"' --network host --shm-size=16g \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 --stop-timeout=600 \
  --memory=112g --memory-swap=112g \
  -v "$MODEL:$MODEL:ro" -v "$ALT:$ALT:ro" -v "$ROOT/cache:$ROOT/cache:rw" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_PP_LAYER_PARTITION="$PARTITION" \
  -e VLLM_CPU_OFFLOAD_GB_PER_RANK="$CPU_OFFLOAD" \
  -e VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=0 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple -e NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" -e NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" -e NCCL_IB_DISABLE=1 \
  -e TMPDIR=/tmp -e PYTHONUNBUFFERED=1 \
  --entrypoint=/usr/local/bin/vllm "$IMAGE" "${ARGS[@]}"

LOG_PATH=$(docker inspect -f '{{.LogPath}}' "$NAME")
echo "[HB] phase=loading deadline_s=$DEADLINE log=$LOG_PATH"
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
    docker logs --tail 220 "$NAME" 2>&1 || true
    fail "balanced DSpark container exited during model load"
  fi
  if curl -fsS --max-time 3 "$BASE/health" >/dev/null 2>&1; then
    MODELS=$(curl -fsS --max-time 5 "$BASE/v1/models" 2>/dev/null || true)
    if [[ "$MODELS" == *deepseek-v4.1-flash-1017* ]]; then
      echo "[HB ${EL}s] phase=health-ready model=deepseek-v4.1-flash-1017"
      printf '%s\n' "$MODELS" > "$REPORT_DIR/dspark-r4-models.json"
      break
    fi
  fi
  LAST=$(docker logs --tail 1 "$NAME" 2>&1 | tr '\r' '\n' | tail -1 | cut -c1-180)
  echo "[HB ${EL}s] phase=loading silence=$((NOW-LAST_CHANGE))s log_bytes=$SIZE last=$LAST"
  if (( EL >= DEADLINE )); then
    docker logs --tail 220 "$NAME" 2>&1 || true
    fail "startup deadline reached"
  fi
  sleep 20
done

if ! timeout --foreground 1800 python3 "$ROOT/verify-pp5-ssd-async.py" \
  --base "$BASE/v1" --model deepseek-v4.1-flash-1017 \
  --out "$REPORT_DIR/dspark-r4-smoke.json"; then
  fail "balanced DSpark correctness smoke test failed"
fi
if ! timeout --foreground 1800 python3 "$ROOT/benchmark-real-text.py" \
  --base "$BASE/v1" --model deepseek-v4.1-flash-1017 \
  --out "$REPORT_DIR/dspark-r4-benchmark.json" --repeats "${BENCHMARK_REPEATS:-2}"; then
  fail "balanced DSpark real-text benchmark failed"
fi
KEEP_RUNNING=1
echo "[HB] phase=RUN_DONE mode=dspark-balanced container=$NAME partition=$PARTITION cpu_offload_gb=$CPU_OFFLOAD_GB cpu_offload_per_rank=$CPU_OFFLOAD api=$BASE/v1"
