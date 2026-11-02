#!/usr/bin/env bash
set -Eeuo pipefail
if [[ ${AIULTRAYANG_AB_GUARD:-0} != 1 ]]; then
  exec timeout --foreground --kill-after=60 2400 env AIULTRAYANG_AB_GUARD=1 bash "$0" "$@"
fi

BASE=deepseek-v41-pp5-trap31-p2p-20261102
CAND=deepseek-v41-aiultrayang-pp5-ssd-matched-20261102
IMAGE=local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-lazy-20261102
ROOT=/run/media/xpz/models/deploy-llm/isolated-aiultrayang-pp5-ssd-async-20261102
MODEL=/run/media/xpz/models/DeepSeek-V4.1-Flash-1017-official
ALT=/home/xpz/.cache/deepseek-v41-official
CACHE=/run/media/xpz/models/deploy-llm/isolated-aiultrayang-pp5-ssd-async-20261031/cache
PORT=18023
BASE_URL=http://127.0.0.1:9010/v1
CAND_URL=http://127.0.0.1:$PORT/v1
START=$(date +%s)
mkdir -p "$ROOT/reports"
exec > >(tee -a "$ROOT/reports/matched-ab.log") 2>&1
phase(){ printf '%s\n' "$1" > "$ROOT/reports/matched-ab.phase"; }
hb(){ while sleep 15; do echo "[HB $(( $(date +%s)-START ))s] phase=$(cat "$ROOT/reports/matched-ab.phase" 2>/dev/null || echo unknown) base=$(docker inspect -f '{{.State.Status}}' "$BASE" 2>/dev/null || echo absent) cand=$(docker inspect -f '{{.State.Status}}' "$CAND" 2>/dev/null || echo absent)"; done; }
hb & HBPID=$!
RESTORE_DONE=0
restore(){
  RC=$?
  phase restore
  docker stop --time 120 "$CAND" >/dev/null 2>&1 || true
  if [[ $(docker inspect -f '{{.State.Status}}' "$BASE" 2>/dev/null || echo absent) != running ]]; then docker start "$BASE" >/dev/null || true; fi
  for _ in $(seq 1 600); do
    if curl -fsS --max-time 2 "$BASE_URL/models" >/dev/null 2>&1; then
      RUNNING=$(curl -fsS --max-time 3 http://127.0.0.1:9010/metrics | awk '/^vllm:num_requests_running\{/ {print $NF; exit}')
      WAITING=$(curl -fsS --max-time 3 http://127.0.0.1:9010/metrics | awk '/^vllm:num_requests_waiting\{/ {print $NF; exit}')
      echo "[HB] phase=restore-done rc=$RC base=$(docker inspect -f '{{.State.Status}}' "$BASE" 2>/dev/null || echo absent) running=$RUNNING waiting=$WAITING"
      RESTORE_DONE=1
      kill "$HBPID" 2>/dev/null || true; wait "$HBPID" 2>/dev/null || true
      return "$RC"
    fi
    sleep 2
  done
  echo "[HB] phase=restore-failed rc=$RC base=$(docker inspect -f '{{.State.Status}}' "$BASE" 2>/dev/null || echo absent)"
  kill "$HBPID" 2>/dev/null || true; wait "$HBPID" 2>/dev/null || true
  return 1
}
trap 'rc=$?; restore || rc=1; exit "$rc"' EXIT
trap 'exit 130' INT TERM
fail(){ phase RUN_FAIL; echo "[HB] phase=RUN_FAIL reason=$*"; exit 1; }

phase preflight
docker image inspect "$IMAGE" >/dev/null || fail "lazy AIO candidate image missing"
[[ $(docker inspect -f '{{.State.Status}}' "$BASE" 2>/dev/null || echo absent) == running ]] || fail "current 9010 service is not running"
[[ -r "$MODEL/config.json" && -r "$MODEL/model-00048-of-00048.safetensors" && -r "$ALT/model-00048-of-00048.safetensors" ]] || fail "checkpoint mount unavailable"
docker inspect "$CAND" >/dev/null 2>&1 && fail "candidate name already exists"
ss -lnt | grep -q ":$PORT " && fail "candidate port is occupied"
METRICS=$(curl -fsS --max-time 5 http://127.0.0.1:9010/metrics)
[[ $(printf '%s\n' "$METRICS" | awk '/^vllm:num_requests_running\{/ {print $NF; exit}') == 0.0 ]] || fail "baseline has active requests"
[[ $(printf '%s\n' "$METRICS" | awk '/^vllm:num_requests_waiting\{/ {print $NF; exit}') == 0.0 ]] || fail "baseline has queued requests"
docker inspect "$BASE" > "$ROOT/reports/baseline-container-inspect.json"
printf '%s\n' "$METRICS" > "$ROOT/reports/baseline-metrics-before.txt"
GPU_UUIDS=$(docker inspect "$BASE" --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)
[[ -n "$GPU_UUIDS" ]] || fail "cannot preserve baseline GPU UUID order"
echo "[HB] phase=preflight baseline=$BASE candidate=$CAND gpu_uuids=$GPU_UUIDS"

phase baseline-serial
timeout --foreground 600 python3 "$ROOT/benchmark-real-text.py" --base "$BASE_URL" --model deepseek-v4.1-flash-1017 --repeats 3 --out "$ROOT/reports/baseline-serial-3x3.json"
phase baseline-concurrency
timeout --foreground 600 python3 "$ROOT/benchmark-pp5-concurrency.py" --base "$BASE_URL" --model deepseek-v4.1-flash-1017 --rounds 10 --out "$ROOT/reports/baseline-concurrency-4x10.json"

phase stop-baseline
docker stop --time 120 "$BASE" >/dev/null
phase candidate-start
docker run -d --init --name "$CAND" \
  --gpus all --network host --shm-size=16g \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 --stop-timeout=600 \
  --memory=112g --memory-swap=112g \
  -v "$MODEL:$MODEL:ro" -v "$ALT:$ALT:ro" -v "$CACHE:$CACHE:rw" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e "CUDA_VISIBLE_DEVICES=$GPU_UUIDS" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_PP_LAYER_PARTITION=8,8,8,8,8 -e VLLM_CPU_OFFLOAD_GB_PER_RANK=0,0,0,0,0 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e VLLM_PP5_LAZY_RECEIVE=1 -e VLLM_PP5_SHARED_MAPPING=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple -e NCCL_P2P_DISABLE=0 -e NCCL_IB_DISABLE=1 -e NCCL_P2P_LEVEL=SYS \
  -e XDG_CACHE_HOME="$CACHE" -e TRITON_CACHE_DIR="$CACHE/triton" -e TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor" \
  -e TMPDIR=/tmp -e XDG_RUNTIME_DIR=/tmp -e PYTHONUNBUFFERED=1 \
  --entrypoint=/usr/local/bin/vllm "$IMAGE" serve "$MODEL" \
  --host=0.0.0.0 --port="$PORT" --served-model-name=deepseek-v4.1-flash-1017 \
  --pipeline-parallel-size=5 --tensor-parallel-size=1 \
  --max-model-len=1048576 --max-num-seqs=8 --max-num-batched-tokens=2048 --gpu-memory-utilization=0.97 \
  --kv-cache-dtype=fp8_ds_mla --trust-remote-code --disable-custom-all-reduce --enable-prefix-caching \
  --safetensors-load-strategy=lazy --engram-config '{"cpu_offload":true,"storage":"ssd","engram_cache_gib":8}' \
  --tokenizer-mode=deepseek_v41 --enable-auto-tool-choice --tool-call-parser=deepseek_v41 --reasoning-parser=deepseek_v41 \
  --enable-prompt-tokens-details \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}'

phase loading
DEADLINE=1500
while true; do
  EL=$(( $(date +%s)-START ))
  [[ $(docker inspect -f '{{.State.Status}}' "$CAND" 2>/dev/null || echo missing) == running ]] || { docker inspect "$CAND" --format 'exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' || true; docker logs --tail 120 "$CAND" 2>&1 || true; fail "candidate exited during load"; }
  if curl -fsS --max-time 3 "$CAND_URL/models" >/dev/null 2>&1; then echo "[HB ${EL}s] phase=health-ready"; break; fi
  (( EL < DEADLINE )) || { docker logs --tail 120 "$CAND" 2>&1 || true; fail "candidate startup deadline reached"; }
  LAST=$(docker logs --tail 1 "$CAND" 2>&1 | tr '\r' '\n' | tail -1 | cut -c1-180)
  echo "[HB ${EL}s] phase=loading last=$LAST"
  sleep 20
done

phase candidate-smoke
timeout --foreground 1800 python3 "$ROOT/verify-pp5-ssd-async.py" --base "$CAND_URL" --model deepseek-v4.1-flash-1017 --out "$ROOT/reports/candidate-smoke.json"
phase candidate-serial
timeout --foreground 600 python3 "$ROOT/benchmark-real-text.py" --base "$CAND_URL" --model deepseek-v4.1-flash-1017 --repeats 3 --out "$ROOT/reports/candidate-serial-3x3.json"
phase candidate-concurrency
timeout --foreground 600 python3 "$ROOT/benchmark-pp5-concurrency.py" --base "$CAND_URL" --model deepseek-v4.1-flash-1017 --rounds 10 --out "$ROOT/reports/candidate-concurrency-4x10.json"
phase compare
timeout --foreground 60 python3 "$ROOT/compare_benchmarks.py" "$ROOT/reports/baseline-serial-3x3.json" "$ROOT/reports/candidate-serial-3x3.json" --out "$ROOT/reports/output-compare.json"
phase TEST_DONE
echo "[HB] phase=TEST_DONE candidate=$CAND"
