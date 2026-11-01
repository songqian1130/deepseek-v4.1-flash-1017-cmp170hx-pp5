#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${ROOT:-/srv/models/deploy-llm/isolated-cpu-hybrid-pp5-20261031}
IMAGE=${IMAGE:-local/deepseek-v41-official-cpu-hybrid-pp5:20261031-r1}
MODEL=/srv/models/DeepSeek-V4.1-Flash-1017-official
ALT=/home/llm/.cache/deepseek-v41-official
NAME=${NAME:-dsv41-cpu-hybrid-pp5-nodspark-20261031}
PORT=${PORT:-18031}
PARTITION=${PARTITION:-5,5,5,4,21}
CPU_START=${CPU_START:-20}
CPU_END=${CPU_END:-40}
BASE=http://127.0.0.1:${PORT}
REPORTS="$ROOT/reports"
START=$(date +%s)
DEADLINE=${STARTUP_DEADLINE:-7200}
KEEP_RUNNING=0
mkdir -p "$REPORTS" "$ROOT/cache"
exec > >(tee -a "$REPORTS/run.log") 2>&1
heartbeat() { while sleep 20; do EL=$(( $(date +%s)-START )); STATE=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo absent); echo "[CPU-HYBRID-HB ${EL}s] phase=monitor container=$STATE pp=$PARTITION port=$PORT"; done; }
heartbeat & HB_PID=$!
cleanup_hb() { kill "$HB_PID" 2>/dev/null || true; wait "$HB_PID" 2>/dev/null || true; }
on_exit() { RC=$?; cleanup_hb; if (( RC != 0 && KEEP_RUNNING == 0 )); then echo "[CPU-HYBRID-HB] phase=cleanup stopping_only=$NAME exit=$RC"; docker stop --time 120 "$NAME" >/dev/null 2>&1 || true; fi; }
trap on_exit EXIT
trap 'exit 130' INT TERM
fail() { echo "[CPU-HYBRID-FAIL] $*"; exit 1; }

echo "[CPU-HYBRID-HB 0s] phase=preflight name=$NAME image=$IMAGE port=$PORT pp=$PARTITION"
if docker inspect "$NAME" >/dev/null 2>&1; then fail "container name exists; refusing replacement"; fi
if ss -lnt | grep -q ":${PORT} "; then fail "port is occupied: $PORT"; fi
docker image inspect "$IMAGE" >/dev/null || fail "image missing: $IMAGE"
for path in "$MODEL/config.json" "$MODEL/model-00048-of-00048.safetensors" "$ALT/model-00048-of-00048.safetensors"; do [[ -r "$path" ]] || fail "checkpoint path unavailable: $path"; done
GPU_APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed '/^[[:space:]]*$/d')
[[ -z "$GPU_APPS" ]] || fail "GPU compute processes are present; no process will be stopped"
ADDITIONAL=$(printf '{"deepseek_v41_hybrid":{"pipeline_layers":[%s],"cpu_expert_start_layer":%s,"cpu_expert_end_layer":%s,"cpu_threads":[8,8],"engram_storage":"ssd","engram_cache_gib":4,"host_cache_gib":0,"overlap_decode":false,"r8_refill":false,"expert_profile":false}}' "$PARTITION" "$CPU_START" "$CPU_END")
ARGS=(serve "$MODEL" --host=127.0.0.1 --port="$PORT" --served-model-name=deepseek-v4.1-flash-1017 --pipeline-parallel-size=5 --tensor-parallel-size=1 --max-model-len=8192 --max-num-seqs=1 --max-num-batched-tokens=1025 --gpu-memory-utilization=0.94 --trust-remote-code --disable-custom-all-reduce --tokenizer-mode=deepseek_v41 --reasoning-parser=deepseek_v41 --enable-prompt-tokens-details --additional-config "$ADDITIONAL")

echo "[CPU-HYBRID-HB] phase=container-start layers=$PARTITION cpu_moe=$CPU_START-$((CPU_END-1)) threads=8 dspark=disabled"
docker run -d --init --name "$NAME" --gpus '"device=0,1,2,3,4"' --network host --shm-size=16g --ulimit memlock=-1 --ulimit nofile=1048576:1048576 --stop-timeout=600 --memory=112g --memory-swap=112g -v "$MODEL:$MODEL:ro" -v "$ALT:$ALT:ro" -v "$ROOT/cache:$ROOT/cache:rw" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=0,1,2,3,4 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_PP_LAYER_PARTITION="$PARTITION" -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 -e TMPDIR=/tmp -e PYTHONUNBUFFERED=1 --entrypoint=/usr/local/bin/vllm "$IMAGE" "${ARGS[@]}"

LOG_PATH=$(docker inspect -f '{{.LogPath}}' "$NAME")
echo "[CPU-HYBRID-HB] phase=loading deadline_s=$DEADLINE log=$LOG_PATH"
while true; do
  NOW=$(date +%s); EL=$((NOW - START)); STATE=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo missing)
  if [[ "$STATE" != running ]]; then docker inspect "$NAME" --format 'exit={{.State.ExitCode}} error={{.State.Error}} oom={{.State.OOMKilled}}' || true; docker logs --tail 220 "$NAME" 2>&1 || true; fail "CPU-MoE container exited during model load"; fi
  if curl -fsS --max-time 3 "$BASE/health" >/dev/null 2>&1; then MODELS=$(curl -fsS --max-time 5 "$BASE/v1/models" 2>/dev/null || true); if [[ "$MODELS" == *deepseek-v4.1-flash-1017* ]]; then printf '%s\n' "$MODELS" > "$REPORTS/models.json"; echo "[CPU-HYBRID-HB ${EL}s] phase=health-ready"; break; fi; fi
  LAST=$(docker logs --tail 1 "$NAME" 2>&1 | tr '\r' '\n' | tail -1 | cut -c1-180); echo "[CPU-HYBRID-HB ${EL}s] phase=loading last=$LAST"
  if (( EL >= DEADLINE )); then docker logs --tail 220 "$NAME" 2>&1 || true; fail "startup deadline reached"; fi
  sleep 20
done
if ! timeout --foreground 1800 python3 "$ROOT/verify-pp5-ssd-async.py" --base "$BASE/v1" --model deepseek-v4.1-flash-1017 --out "$REPORTS/smoke.json"; then fail "CPU-MoE smoke test failed"; fi
KEEP_RUNNING=1
echo "[CPU-HYBRID-DONE] container=$NAME pp=$PARTITION cpu_moe=$CPU_START-$((CPU_END-1)) dspark=disabled api=$BASE/v1"
