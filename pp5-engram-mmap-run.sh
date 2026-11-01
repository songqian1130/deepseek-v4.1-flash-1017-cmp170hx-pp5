#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/srv/models/deploy-llm/isolated-pp5-engram-mmap-20261031/pp5-original-9010-r4-reference-aligned}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/cache}
IMAGE=${IMAGE:-local/deepseek-v41-official-pp5:alex-engram-mmap-20261031-r4-reference-aligned}
MODEL=/srv/models/DeepSeek-V4.1-Flash-1017-official
NAME=${NAME:-deepseek-v41-pp5-alex-engram-mmap-20261031-original-9010-r4}
# Match the executable defaults of the reference's 9010 launcher.
# Disk row gathering requires host work between captured graph segments.
CUDA_GRAPH_MODE=${CUDA_GRAPH_MODE:-PIECEWISE}
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
CPU_OFFLOAD_PER_RANK=${CPU_OFFLOAD_PER_RANK:-0,0,0,0,0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1048576}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.97}
KV_CACHE_MEMORY=${KV_CACHE_MEMORY:-0}
NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-5}
SPEC_MODE=${SPEC_MODE:-plain}
LONG_PREFILL_TOKEN_THRESHOLD=${LONG_PREFILL_TOKEN_THRESHOLD:-0}
CG_CAPTURE_SIZES=${CG_CAPTURE_SIZES:-1,2,4,8}
MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-8}
PORT=${PORT:-9010}
HOST=${HOST:-127.0.0.1}
BASE=http://127.0.0.1:${PORT}

mkdir -p "$ROOT/reports" "$CACHE_ROOT"
LOG="$ROOT/reports/run.log"
PHASE_FILE="$ROOT/reports/run.phase"
exec > >(tee -a "$LOG") 2>&1
START=$(date +%s)
KEEP_RUNNING=0
CONTAINER_CREATED=0
phase() { printf '%s\n' "$1" > "$PHASE_FILE"; }
heartbeat() {
  while sleep 20; do
    EL=$(( $(date +%s) - START ))
    STATE=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo absent)
    echo "[HB ${EL}s] phase=$(cat "$PHASE_FILE" 2>/dev/null || echo unknown) container=$STATE"
  done
}
heartbeat & HB_PID=$!
cleanup_hb() { kill "$HB_PID" 2>/dev/null || true; wait "$HB_PID" 2>/dev/null || true; }
stop_own_container() { docker stop --time 120 "$NAME" >/dev/null 2>&1 || true; }
on_exit() {
  RC=$?
  cleanup_hb
  if (( RC != 0 && KEEP_RUNNING == 0 && CONTAINER_CREATED == 1 )); then
    echo "[HB] phase=cleanup stopping_only=$NAME exit=$RC"
    stop_own_container
  fi
}
trap on_exit EXIT
trap 'exit 130' INT TERM
fail() { phase RUN_FAIL; echo "[HB] phase=RUN_FAIL reason=$*"; exit 1; }

phase preflight
echo "[HB 0s] phase=preflight image=$IMAGE port=$PORT pp=5 layers=8,8,8,8,8 cudagraph=$CUDA_GRAPH_MODE spec=$SPEC_MODE"
docker image inspect "$IMAGE" >/dev/null || fail "isolated mmap image is not built"
[[ -r "$MODEL/config.json" && -r "$MODEL/model.safetensors.index.json" ]] || fail "official checkpoint is unavailable"
[[ -r "$MODEL/model-00047-of-00048.safetensors" && -r "$MODEL/model-00048-of-00048.safetensors" ]] || fail "Engram checkpoint shards are unavailable"
if docker inspect "$NAME" >/dev/null 2>&1; then fail "container name already exists; refusing to replace it: $NAME"; fi
if ss -lnt | grep -q ":${PORT} "; then fail "port ${PORT} is already occupied"; fi
GPU_APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed '/^[[:space:]]*$/d')
[[ -z "$GPU_APPS" ]] || fail "GPU compute processes are present; no existing process will be stopped"
case "$CUDA_GRAPH_MODE" in
  NONE) GRAPH_CONFIG='{"cudagraph_mode":"NONE"}' ;;
  PIECEWISE) GRAPH_CONFIG="{\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[${CG_CAPTURE_SIZES}],\"max_cudagraph_capture_size\":${MAX_CUDAGRAPH_CAPTURE_SIZE}}" ;;
  FULL_AND_PIECEWISE)
    [[ "${VLLM_USE_V2_MODEL_RUNNER:-}" == 0 ]] || fail "validated FULL profile requires VLLM_USE_V2_MODEL_RUNNER=0"
    timeout --foreground 30 docker run --rm --network host --entrypoint python3 "$IMAGE" -c '
import hashlib,json,pathlib
root=pathlib.Path("/opt/full-host-break")
manifest=json.loads((root/"alignment-manifest.json").read_text())
proof=json.loads((root/"full-host-break-verification.json").read_text())
verified=json.loads((root/"verified.json").read_text())
revision=manifest["approved_adapter_revision"]
helper=pathlib.Path("/usr/local/lib/python3.12/dist-packages")/revision["path"]
assert proof["ok"] and verified["ok"]
assert hashlib.sha256(helper.read_bytes()).hexdigest()==revision["r6_sha256"]
print("FULL_HOST_BREAK_IMAGE_VERIFIED")
' || fail "FULL needs the verified r6 host-break image"
    GRAPH_CONFIG="{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${CG_CAPTURE_SIZES}],\"max_cudagraph_capture_size\":${MAX_CUDAGRAPH_CAPTURE_SIZE}}"
    ;;
  *) fail "unsupported CUDA graph mode: $CUDA_GRAPH_MODE" ;;
esac
RUNNER_ENV=()
if [[ -n "${VLLM_USE_V2_MODEL_RUNNER:-}" ]]; then
  RUNNER_ENV=(-e "VLLM_USE_V2_MODEL_RUNNER=$VLLM_USE_V2_MODEL_RUNNER")
fi
if [[ -n "${VLLM_ENGRAM_MMAP_CHUNK_ROWS:-}" ]]; then
  RUNNER_ENV+=(-e "VLLM_ENGRAM_MMAP_CHUNK_ROWS=$VLLM_ENGRAM_MMAP_CHUNK_ROWS")
fi

phase real-checkpoint
timeout --foreground 90 docker run --rm --network host \
  -v "$MODEL:$MODEL:ro" \
  -v /home/llm/.cache/deepseek-v41-official:/home/llm/.cache/deepseek-v41-official:ro \
  --entrypoint=python3 "$IMAGE" /opt/verify-real-engram-mmap.py \
  --model-path "$MODEL" --sample-count 5 --workers 8

ARGS=(
  serve "$MODEL"
  --host="$HOST" --port="$PORT"
  --served-model-name=deepseek-v4.1-flash-1017
  --pipeline-parallel-size=5 --tensor-parallel-size=1
  --max-model-len="$MAX_MODEL_LEN" --max-num-seqs="$MAX_NUM_SEQS" --max-num-batched-tokens="$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization="$GPU_MEMORY_UTILIZATION"
  --kv-cache-dtype=fp8_ds_mla
  --trust-remote-code --disable-custom-all-reduce --enable-prefix-caching
  --safetensors-load-strategy=lazy
  --engram-config '{"cpu_offload":true}'
  --tokenizer-mode=deepseek_v41
  --enable-auto-tool-choice --tool-call-parser=deepseek_v41 --reasoning-parser=deepseek_v41
  --enable-prompt-tokens-details
  --compilation-config "$GRAPH_CONFIG"
)
if [[ -n "$KV_CACHE_MEMORY" && "$KV_CACHE_MEMORY" != "0" ]]; then
  ARGS+=(--kv-cache-memory="$KV_CACHE_MEMORY")
fi
if (( CPU_OFFLOAD_GB > 0 )); then
  ARGS+=(--cpu-offload-gb="$CPU_OFFLOAD_GB" --cpu-offload-params experts)
fi
if (( LONG_PREFILL_TOKEN_THRESHOLD > 0 )); then
  ARGS+=(--long-prefill-token-threshold="$LONG_PREFILL_TOKEN_THRESHOLD")
fi
case "$SPEC_MODE" in
  plain) ;;
  dspark) ARGS+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS},\"use_local_argmax_reduction\":true}") ;;
  *) fail "unsupported speculative mode: $SPEC_MODE" ;;
esac

phase container-start
echo "[HB] phase=container-start name=$NAME"
docker run -d --init --name "$NAME" \
  --gpus '"device=0,1,2,3,4"' --network host --shm-size=16g \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 --stop-timeout=600 \
  --memory=112g --memory-swap=112g \
  -v "$MODEL:$MODEL:ro" \
  -v /home/llm/.cache/deepseek-v41-official:/home/llm/.cache/deepseek-v41-official:ro \
  -v "$CACHE_ROOT:$CACHE_ROOT:rw" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_PP_LAYER_PARTITION=8,8,8,8,8 \
  -e VLLM_CPU_OFFLOAD_GB_PER_RANK="$CPU_OFFLOAD_PER_RANK" \
  -e DS41_ENGRAM_MANIFEST="$MODEL/model.safetensors.index.json" \
  -e VLLM_ENGRAM_MMAP=1 -e VLLM_ENGRAM_MMAP_WORKERS=16 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  "${RUNNER_ENV[@]}" \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 \
  -e XDG_CACHE_HOME="$CACHE_ROOT" -e TRITON_CACHE_DIR="$CACHE_ROOT/triton" \
  -e TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 \
  -e TMPDIR=/tmp -e PYTHONUNBUFFERED=1 \
  --entrypoint=/usr/local/bin/vllm "$IMAGE" "${ARGS[@]}"
CONTAINER_CREATED=1

LOG_PATH=$(docker inspect -f '{{.LogPath}}' "$NAME")
phase loading
echo "[HB] phase=loading deadline_s=7200 log=$LOG_PATH"
DEADLINE=7200
while true; do
  NOW=$(date +%s)
  EL=$((NOW - START))
  STATE=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo missing)
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
      printf '%s\n' "$MODELS" > "$ROOT/reports/models.json"
      break
    fi
  fi
  if (( EL >= DEADLINE )); then
    echo "[HB ${EL}s] phase=deadline-stop-only container=$NAME"
    docker logs --tail 120 "$NAME" 2>&1 || true
    fail "120-minute startup deadline reached"
  fi
  LAST=$(docker logs --tail 1 "$NAME" 2>&1 | tr '\r' '\n' | tail -1 | cut -c1-180)
  LOG_BYTES=$(stat -c %s "$LOG_PATH" 2>/dev/null || echo 0)
  echo "[HB ${EL}s] phase=loading log_bytes=$LOG_BYTES last=$LAST"
  sleep 20
done

phase smoke
timeout --foreground 1800 python3 "$(dirname -- "${BASH_SOURCE[0]}")/verify-pp5-ssd-async.py" \
  --base "$BASE/v1" --model deepseek-v4.1-flash-1017 \
  --out "$ROOT/reports/smoke.json"
timeout --foreground 1800 python3 "$(dirname -- "${BASH_SOURCE[0]}")/benchmark-real-text.py" \
  --base "$BASE/v1" --model deepseek-v4.1-flash-1017 --repeats 1 \
  --out "$ROOT/reports/benchmark.json"
KEEP_RUNNING=1
phase RUN_DONE
echo "[HB] phase=RUN_DONE container=$NAME api=$BASE/v1 reports=$ROOT/reports"
