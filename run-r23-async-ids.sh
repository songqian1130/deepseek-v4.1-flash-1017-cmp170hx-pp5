#!/usr/bin/env bash
set -Eeuo pipefail

BASE=deepseek-v41-pp5-alex-engram-mmap-20261101-r10-lan
CAND=deepseek-v41-pp5-alex-engram-mmap-20261101-r23-async-ids
ROOT=/srv/models/deploy-llm/isolated-pp5-engram-mmap-20261101/r23-async-ids
IMAGE=local/deepseek-v41-official-pp5:async-engram-ids-r1-full
RUNNER=/srv/models/deploy-llm/isolated-pp5-engram-mmap-20261031/start_vllm_v41_pp5_full.sh
PORT=9027
START=$(date +%s)
mkdir -p "$ROOT/reports"
exec > >(tee -a "$ROOT/reports/wrapper.log") 2>&1
phase(){ printf '%s\n' "$1" > "$ROOT/reports/phase"; }
hb(){ while sleep 20; do echo "[HB $(( $(date +%s)-START ))s] phase=$(cat "$ROOT/reports/phase" 2>/dev/null || echo unknown) base=$(docker inspect -f '{{.State.Status}}' "$BASE" 2>/dev/null || echo absent) cand=$(docker inspect -f '{{.State.Status}}' "$CAND" 2>/dev/null || echo absent)"; done; }
hb & HBPID=$!
restore(){ RC=$?; phase restore; docker stop --time 120 "$CAND" >/dev/null 2>&1 || true; if [[ $(docker inspect -f '{{.State.Status}}' "$BASE" 2>/dev/null || echo absent) != running ]]; then docker start "$BASE" >/dev/null || true; for _ in $(seq 1 180); do curl -fsS --max-time 2 http://127.0.0.1:9010/health >/dev/null 2>&1 && break; sleep 2; done; fi; echo "[HB] phase=restore-done rc=$RC base=$(docker inspect -f '{{.State.Status}}' "$BASE" 2>/dev/null || echo absent)"; kill "$HBPID" 2>/dev/null || true; wait "$HBPID" 2>/dev/null || true; exit "$RC"; }
trap restore EXIT
phase preflight
docker image inspect "$IMAGE" >/dev/null
[[ $(docker inspect -f '{{.State.Status}}' "$BASE" 2>/dev/null || echo absent) == running ]]
if docker inspect "$CAND" >/dev/null 2>&1; then echo candidate exists >&2; exit 1; fi
if ss -lnt | grep -q ":$PORT "; then echo port occupied >&2; exit 1; fi
phase stop-baseline
docker stop --time 120 "$BASE" >/dev/null
phase candidate-start
timeout --foreground 1800 env ROOT="$ROOT" IMAGE="$IMAGE" NAME="$CAND" PORT="$PORT" HOST=127.0.0.1 CUDA_GRAPH_MODE=FULL_AND_PIECEWISE VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENGRAM_MMAP_CHUNK_ROWS=4 VLLM_PP5_LAZY_RECEIVE=1 VLLM_PP5_SHARED_MAPPING=1 SPEC_MODE=plain CPU_OFFLOAD_GB=0 CPU_OFFLOAD_PER_RANK=0,0,0,0,0 bash "$RUNNER" --plain
phase benchmark
timeout --foreground 300 python3 /tmp/benchmark-real-text.py --base "http://127.0.0.1:$PORT/v1" --model deepseek-v4.1-flash-1017 --repeats 2 --out "$ROOT/reports/benchmark-2x3.json"
phase done
echo "[HB] phase=RUN_DONE candidate=$CAND"
