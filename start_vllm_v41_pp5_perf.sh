#!/usr/bin/env bash
set -Eeuo pipefail
# Same model, quantization and 9010 PP5 parameters; scheduling and SSD reads.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export ROOT=${ROOT:-/srv/models/deploy-llm/isolated-pp5-engram-mmap-20261101/r10-production}
export IMAGE=${IMAGE:-local/deepseek-v41-official-pp5:alex-engram-mmap-20261101-r8-perf}
export NAME=${NAME:-deepseek-v41-pp5-alex-engram-mmap-20261101-r10-lan}
export VLLM_ENGRAM_MMAP_CHUNK_ROWS=${VLLM_ENGRAM_MMAP_CHUNK_ROWS:-4}
export HOST=${HOST:-0.0.0.0}
timeout --foreground 30 docker run --rm --network host --entrypoint python3 "$IMAGE" -c '
import hashlib,json,pathlib
root=pathlib.Path("/opt/pp5-performance-source")
manifest=json.loads((root/"alignment-manifest.json").read_text())
assert json.loads((root/"verified.json").read_text())["ok"]
for name,revision in manifest["approved_performance_revision"]["files"].items():
    source=pathlib.Path("/usr/local/lib/python3.12/dist-packages")/name
    assert hashlib.sha256(source.read_bytes()).hexdigest()==revision["sha256"]
key=manifest["approved_performance_revision"]["new_helper"]
source=pathlib.Path("/usr/local/lib/python3.12/dist-packages")/key
assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest["extra_hashes"][key]
print("PP5_PERFORMANCE_IMAGE_VERIFIED")
'
export VLLM_USE_V2_MODEL_RUNNER=0
export CUDA_GRAPH_MODE=FULL_AND_PIECEWISE
exec bash "$SCRIPT_DIR/start_vllm_v41_pp5_full.sh" "$@"
