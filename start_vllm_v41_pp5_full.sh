#!/usr/bin/env bash
set -Eeuo pipefail

# Validated 2026-11-01: original 9010 PP5 defaults, SSD host breaks,
# Runner V1 and FULL decode. Keep the r4 launcher available for rollback.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export ROOT=${ROOT:-/srv/models/deploy-llm/isolated-pp5-engram-mmap-20261101/r10-production}
export IMAGE=${IMAGE:-local/deepseek-v41-official-pp5:alex-engram-mmap-20261101-r8-perf}
export NAME=${NAME:-deepseek-v41-pp5-alex-engram-mmap-20261101-r10-lan}
export CUDA_GRAPH_MODE=FULL_AND_PIECEWISE
export VLLM_USE_V2_MODEL_RUNNER=0
export HOST=${HOST:-0.0.0.0}
export VLLM_ENGRAM_MMAP_CHUNK_ROWS=${VLLM_ENGRAM_MMAP_CHUNK_ROWS:-4}
exec bash "$SCRIPT_DIR/start_vllm_v41_pp5.sh" "$@"
