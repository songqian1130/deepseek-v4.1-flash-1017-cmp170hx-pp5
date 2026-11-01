#!/usr/bin/env bash
set -Eeuo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$HERE}
BUILD_PID=${BUILD_PID:?Set BUILD_PID to the bounded r3 build launcher PID}
BUILD_PHASE="$ROOT/reports/pp5-graph-build.phase"
START=$(date +%s)

while [[ $(cat "$BUILD_PHASE" 2>/dev/null || true) != BUILD_DONE ]]; do
  EL=$(( $(date +%s) - START ))
  PHASE=$(cat "$BUILD_PHASE" 2>/dev/null || echo unknown)
  echo "[HB ${EL}s] phase=waiting-for-r3-build build_phase=$PHASE"
  if [[ "$PHASE" == BUILD_FAIL ]] || ! kill -0 "$BUILD_PID" 2>/dev/null; then
    echo "RUN_AFTER_BUILD_FAIL: builder exited without BUILD_DONE"
    exit 1
  fi
  if (( EL >= 1200 )); then
    echo "RUN_AFTER_BUILD_FAIL: finite build wait deadline reached"
    exit 124
  fi
  sleep 20
done

echo "[HB] phase=build-verified-starting-9010"
exec timeout --foreground 10800 env \
  ROOT="$ROOT" \
  IMAGE=local/deepseek-v41-official-pp5:alex-engram-mmap-20261031-r3-pp-capture-ids \
  NAME=deepseek-v41-pp5-alex-engram-mmap-20261031-original-9010-r3 \
  PORT=9010 CUDA_GRAPH_MODE=PIECEWISE \
  CPU_OFFLOAD_GB=0 CPU_OFFLOAD_PER_RANK=0,0,0,0,0 \
  MAX_MODEL_LEN=1048576 MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=8 \
  GPU_MEMORY_UTILIZATION=0.97 KV_CACHE_MEMORY=0 SPEC_MODE=plain \
  LONG_PREFILL_TOKEN_THRESHOLD=0 \
  CG_CAPTURE_SIZES=1,2,4,8 MAX_CUDAGRAPH_CAPTURE_SIZE=8 \
  bash "$HERE/start_vllm_v41_pp5.sh"
