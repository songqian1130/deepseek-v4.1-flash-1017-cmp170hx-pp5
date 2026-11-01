#!/usr/bin/env bash
set -Eeuo pipefail
BASE_NAME=deepseek-v41-pp5-alex-engram-mmap-20261101-r10-lan
CAND_NAME=deepseek-v41-pp5-alex-engram-mmap-20261101-r22-partition-78898
ROOT=/srv/models/deploy-llm/isolated-pp5-engram-mmap-20261101/r22-partition-78898
IMAGE=local/deepseek-v41-official-pp5:alex-engram-mmap-20261101-r8-perf
RUNNER=/srv/models/deploy-llm/isolated-pp5-engram-mmap-20261031/pp5-engram-mmap-run.sh
BACKUP=$ROOT/runner.backup
PARTITION=7,8,8,9,8
PORT=9026
START=$(date +%s)
mkdir -p "$ROOT/reports"
exec > >(tee -a "$ROOT/reports/wrapper.log") 2>&1
phase(){ printf '%s\n' "$1" > "$ROOT/reports/phase"; }
hb(){ while sleep 20; do echo "[HB $(( $(date +%s)-START ))s] phase=$(cat "$ROOT/reports/phase" 2>/dev/null || echo unknown) base=$(docker inspect -f '{{.State.Status}}' "$BASE_NAME" 2>/dev/null || echo absent) cand=$(docker inspect -f '{{.State.Status}}' "$CAND_NAME" 2>/dev/null || echo absent)"; done; }
hb & HBPID=$!
restore(){ RC=$?; phase restore; docker stop --time 120 "$CAND_NAME" >/dev/null 2>&1 || true; if [[ -f "$BACKUP" ]]; then cp "$BACKUP" "$RUNNER"; fi; if [[ $(docker inspect -f '{{.State.Status}}' "$BASE_NAME" 2>/dev/null || echo absent) != running ]]; then docker start "$BASE_NAME" >/dev/null || true; for _ in $(seq 1 180); do curl -fsS --max-time 2 http://127.0.0.1:9010/health >/dev/null 2>&1 && break; sleep 2; done; fi; echo "[HB] phase=restore-done rc=$RC base=$(docker inspect -f '{{.State.Status}}' "$BASE_NAME" 2>/dev/null || echo absent)"; kill "$HBPID" 2>/dev/null || true; wait "$HBPID" 2>/dev/null || true; exit "$RC"; }
trap restore EXIT
phase preflight
docker image inspect "$IMAGE" >/dev/null
[[ $(docker inspect -f '{{.State.Status}}' "$BASE_NAME" 2>/dev/null || echo absent) == running ]]
if docker inspect "$CAND_NAME" >/dev/null 2>&1; then echo candidate exists >&2; exit 1; fi
cp "$RUNNER" "$BACKUP"
python3 - "$RUNNER" "$PARTITION" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); part=sys.argv[2]; s=p.read_text()
old='  -e VLLM_PP_LAYER_PARTITION=8,8,8,8,8 \\\n'
new='  -e VLLM_PP_LAYER_PARTITION="$PP_LAYER_PARTITION" \\\n'
assert old in s, 'runner partition line not found'
p.write_text(s.replace(old,new,1))
PY
phase stop-baseline
docker stop --time 120 "$BASE_NAME" >/dev/null
phase run
export PP_LAYER_PARTITION="$PARTITION"
timeout --foreground 1500 env ROOT="$ROOT" IMAGE="$IMAGE" NAME="$CAND_NAME" PORT="$PORT" HOST=127.0.0.1 CUDA_GRAPH_MODE=FULL_AND_PIECEWISE VLLM_USE_V2_MODEL_RUNNER=0 VLLM_PP5_LAZY_RECEIVE=1 VLLM_PP5_SHARED_MAPPING=1 VLLM_ENGRAM_MMAP_CHUNK_ROWS=4 SPEC_MODE=plain CPU_OFFLOAD_GB=0 CPU_OFFLOAD_PER_RANK=0,0,0,0,0 bash "$RUNNER"
phase candidate-benchmark
python3 /tmp/benchmark-real-text.py --base "http://127.0.0.1:${PORT}/v1" --model deepseek-v4.1-flash-1017 --repeats 2 --out "$ROOT/reports/benchmark-2x3.json"
phase candidate-quality
python3 /tmp/verify-pp5-no-degradation.py --base "http://127.0.0.1:${PORT}" --out "$ROOT/reports/quality-parity" --reference-dir /srv/models/deploy-llm/isolated-pp5-engram-mmap-20261031/performance-hotspots-20261101/r9-production/quality-parity
phase done
echo "[HB] phase=RUN_DONE partition=$PARTITION candidate=$CAND_NAME"