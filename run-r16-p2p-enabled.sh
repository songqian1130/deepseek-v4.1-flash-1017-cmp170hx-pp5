#!/usr/bin/env bash
set -Eeuo pipefail
BASE_NAME=deepseek-v41-pp5-alex-engram-mmap-20261101-r10-lan
CAND_NAME=deepseek-v41-pp5-alex-engram-mmap-20261101-r16-p2p-enabled
ROOT=/srv/models/deploy-llm/isolated-pp5-r16-p2p-enabled
IMAGE=local/deepseek-v41-official-pp5:alex-engram-mmap-20261101-r8-perf
PORT=9024
RUNNER=/srv/models/deploy-llm/isolated-pp5-engram-mmap-20261031/pp5-engram-mmap-run.sh
BACKUP=$ROOT/runner.backup
mkdir -p "$ROOT/reports"
LOG=$ROOT/reports/wrapper.log
exec > >(tee -a "$LOG") 2>&1
START=$(date +%s)
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
python3 - "$RUNNER" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); s=p.read_text(); a='-e NCCL_P2P_DISABLE=1'; assert a in s; p.write_text(s.replace(a, '-e NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE"', 1))
PY
phase stop-baseline
docker stop --time 120 "$BASE_NAME" >/dev/null
phase run
export NCCL_P2P_DISABLE=0
timeout --foreground 1800 env ROOT="$ROOT" IMAGE="$IMAGE" NAME="$CAND_NAME" PORT="$PORT" HOST=127.0.0.1 CUDA_GRAPH_MODE=FULL_AND_PIECEWISE VLLM_USE_V2_MODEL_RUNNER=0 VLLM_PP5_LAZY_RECEIVE=1 VLLM_PP5_SHARED_MAPPING=1 VLLM_ENGRAM_MMAP_CHUNK_ROWS=4 SPEC_MODE=plain CPU_OFFLOAD_GB=0 CPU_OFFLOAD_PER_RANK=0,0,0,0,0 bash "$RUNNER"
phase done
echo "[HB] phase=RUN_DONE candidate=$CAND_NAME"
