#!/usr/bin/env bash
set -Eeuo pipefail

# Reference: v41-pp5-1m-pinmem-20260916, 9010 original launcher.
# pp5-engram-mmap-run.sh supplies the same PP5 inference defaults,
# with PIECEWISE graphs for the original SSD backend. The verified r6
# FULL profile has its own start_vllm_v41_pp5_full.sh entry point.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export ROOT=${ROOT:-/srv/models/deploy-llm/isolated-pp5-engram-mmap-20261031/pp5-original-9010-r4-reference-aligned}
while (($#)); do
  case "$1" in
    --port|--maxlen|--gpu-util|--max-batched|--max-seqs)
      (($# >= 2)) || { echo "missing value for $1" >&2; exit 2; }
      case "$1" in
        --port) export PORT=$2 ;;
        --maxlen) export MAX_MODEL_LEN=$2 ;;
        --gpu-util) export GPU_MEMORY_UTILIZATION=$2 ;;
        --max-batched) export MAX_NUM_BATCHED_TOKENS=$2 ;;
        --max-seqs) export MAX_NUM_SEQS=$2 ;;
      esac
      shift 2
      ;;
    --piecewise) export CUDA_GRAPH_MODE=PIECEWISE; shift ;;
    --eager) export CUDA_GRAPH_MODE=NONE; shift ;;
    --plain) export SPEC_MODE=plain; shift ;;
    --dspark)
      export SPEC_MODE=dspark
      shift
      ;;
    -h|--help)
      printf '%s\n' \
        'PP5 9010 reference profile with SSD Engram adaptation.' \
        'Defaults: 8,8,8,8,8; 1M context; util 0.97; batch 2048; 8 seq; plain.' \
        'Options: --port P --maxlen N --gpu-util N --max-batched N --max-seqs N' \
        '         --piecewise --eager --plain --dspark' \
        'Original SSD profile: PIECEWISE or NONE.' \
        'Verified V1 + FULL profile: start_vllm_v41_pp5_full.sh (r6 image).'
      exit 0
      ;;
    *) echo "unsupported option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
if [[ "${SPEC_MODE:-plain}" == dspark ]]; then
  export CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-9}
  export CPU_OFFLOAD_PER_RANK=${CPU_OFFLOAD_PER_RANK:-0,0,0,0,9}
fi
exec bash "$SCRIPT_DIR/pp5-engram-mmap-run.sh"
