#!/usr/bin/env bash
set -Eeuo pipefail

BASE_NAME=deepseek-v41-pp5-alex-engram-mmap-20261101-r10-lan
DS_NAME=deepseek-v41-pp5-alex-engram-mmap-20261101-r11-dspark-v2-only
DS_ROOT=/srv/models/deploy-llm/isolated-pp5-dspark-r11-20261101
DS_IMAGE=local/deepseek-v41-official-pp5:alex-engram-mmap-20261101-r11-dspark-construction-uva
PORT=9020
RUNNER=/srv/models/deploy-llm/isolated-pp5-engram-mmap-20261031/pp5-engram-mmap-run.sh
BACKUP=$DS_ROOT/pp5-engram-mmap-run.sh.v2-backup
PHASE_FILE=$DS_ROOT/reports/dspark-v2-only.phase
LOG_FILE=$DS_ROOT/reports/dspark-v2-only-wrapper.log
mkdir -p "$DS_ROOT/reports"
exec > >(tee -a "$LOG_FILE") 2>&1
phase() { printf '%s\n' "$1" > "$PHASE_FILE"; }
heartbeat() {
  while sleep 20; do
    echo "[HB $(date +%s)] phase=$(cat "$PHASE_FILE" 2>/dev/null || echo unknown) baseline=$(docker inspect -f '{{.State.Status}}' "$BASE_NAME" 2>/dev/null || echo absent) dspark=$(docker inspect -f '{{.State.Status}}' "$DS_NAME" 2>/dev/null || echo absent)"
  done
}
heartbeat & HB_PID=$!
restore() {
  RC=$?
  phase restore
  docker stop --time 120 "$DS_NAME" >/dev/null 2>&1 || true
  if [[ -f "$BACKUP" ]]; then cp "$BACKUP" "$RUNNER"; fi
  if [[ $(docker inspect -f '{{.State.Status}}' "$BASE_NAME" 2>/dev/null || echo absent) != running ]]; then
    docker start "$BASE_NAME" >/dev/null
    for _ in $(seq 1 120); do
      curl -fsS --max-time 2 http://127.0.0.1:9010/health >/dev/null 2>&1 && break
      sleep 2
    done
  fi
  echo "[HB] phase=restore-done rc=$RC baseline=$(docker inspect -f '{{.State.Status}}' "$BASE_NAME" 2>/dev/null || echo absent)"
  kill "$HB_PID" 2>/dev/null || true
  wait "$HB_PID" 2>/dev/null || true
  exit "$RC"
}
trap restore EXIT

phase preflight
docker image inspect "$DS_IMAGE" >/dev/null
[[ $(docker inspect -f '{{.State.Status}}' "$BASE_NAME" 2>/dev/null || echo absent) == running ]]
if docker inspect "$DS_NAME" >/dev/null 2>&1; then exit 1; fi
cp "$RUNNER" "$BACKUP"
python3 - "$RUNNER" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
anchor = '  -e VLLM_CPU_OFFLOAD_GB_PER_RANK="$CPU_OFFLOAD_PER_RANK" \\\n'
insert = anchor + '  -e VLLM_DSPARK_CPU_OFFLOAD_GB="$VLLM_DSPARK_CPU_OFFLOAD_GB" \\\n  -e VLLM_DSPARK_CONSTRUCTION_UVA="$VLLM_DSPARK_CONSTRUCTION_UVA" \\\n  -e VLLM_PP5_LAZY_RECEIVE=0 \\\n  -e VLLM_PP5_SHARED_MAPPING=0 \\\n'
if anchor not in s:
    raise SystemExit("runner env anchor not found")
p.write_text(s.replace(anchor, insert, 1))
PY
phase stop-baseline
docker stop --time 120 "$BASE_NAME" >/dev/null
phase dspark-v2-run
export VLLM_DSPARK_CPU_OFFLOAD_GB=9
export VLLM_DSPARK_CONSTRUCTION_UVA=1
timeout --foreground 1800 env ROOT="$DS_ROOT" IMAGE="$DS_IMAGE" NAME="$DS_NAME" PORT="$PORT" HOST=127.0.0.1 CUDA_GRAPH_MODE=PIECEWISE VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENGRAM_MMAP_CHUNK_ROWS=4 SPEC_MODE=dspark CPU_OFFLOAD_GB=0 CPU_OFFLOAD_PER_RANK=0,0,0,0,0 bash "$RUNNER"
phase dspark-v2-done
echo "[HB] phase=RUN_DONE dspark=$DS_NAME"
