#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${ROOT:-/srv/models/deploy-llm/isolated-cpu-hybrid-pp5-20261031}
NATIVE_ROOT=${NATIVE_ROOT:-/srv/models/deploy-llm/isolated-cpu-moe-pp5-20261031}
REF=${REF:-/srv/models/deploy-llm/reference-buliaoyin-cpu-hybrid-20261031/src}
BASE_IMAGE=${BASE_IMAGE:-local/deepseek-v41-official-cpu-hybrid-pp5:20261031-r3}
IMAGE=${IMAGE:-local/deepseek-v41-official-cpu-hybrid-pp5:20261031-r4}
OUT="$ROOT/native-ssd-build-r2"
TOOLS="$NATIVE_ROOT/container-build/pyuser/lib/python3.12/site-packages/cmake"
IK_SOURCE="$NATIVE_ROOT/container-build/work/sources/ik"
CONTEXT="$ROOT/native-image-context"
CONTAINERFILE="$CONTEXT/Containerfile"
START=$(date +%s)
mkdir -p "$ROOT/reports"
exec > >(tee -a "$ROOT/reports/native-build.log") 2>&1
phase() { printf '%s\n' "$1" > "$ROOT/reports/native-build.phase"; }
heartbeat() { while sleep 20; do echo "[NATIVE-BUILD-HB $(( $(date +%s)-START ))s] phase=$(cat "$ROOT/reports/native-build.phase" 2>/dev/null || echo unknown)"; done; }
heartbeat & HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null || true; wait "$HB_PID" 2>/dev/null || true' EXIT
fail() { phase BUILD_FAIL; echo "[NATIVE-BUILD-FAIL] $*"; exit 1; }

phase preflight
docker image inspect "$BASE_IMAGE" >/dev/null || fail "base image missing: $BASE_IMAGE"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then fail "image tag exists; choose a new IMAGE tag"; fi
for path in "$REF/csrc/cpu/dsv41/CMakeLists.txt" "$REF/csrc/cpu/dsv41/engram_ssd.cpp" "$TOOLS/data/bin/cmake" "$IK_SOURCE/.dsv41-source.json"; do test -r "$path" || fail "required build input missing: $path"; done
if test -e "$OUT"; then fail "isolated build output already exists: $OUT"; fi
mkdir -p "$OUT" "$CONTEXT"
phase prepare-source
cp -a "$IK_SOURCE" "$OUT/ik-src"

phase configure
echo "[NATIVE-BUILD-HB 0s] phase=configure output=$OUT/cmake"
timeout --foreground 1800 docker run --rm --network none \
  -v "$REF:$REF:ro" -v "$OUT:$OUT:rw" -v "$TOOLS:/opt/cmake:ro" \
  -v /usr/bin/git:/usr/local/bin/git:ro \
  --entrypoint bash "$BASE_IMAGE" -lc \
  'export PATH=/usr/local/bin:/usr/local/cuda/bin:$PATH CUDAToolkit_ROOT=/usr/local/cuda; /opt/cmake/data/bin/cmake -G Ninja -S "$1/csrc/cpu/dsv41" -B "$2/cmake" -DFETCHCONTENT_SOURCE_DIR_DSV41_IK="$2/ik-src"' _ "$REF" "$OUT"

phase build
echo "[NATIVE-BUILD-HB] phase=build target=libdsv41_cuda"
timeout --foreground 1800 docker run --rm --network none \
  -v "$REF:$REF:ro" -v "$OUT:$OUT:rw" -v "$TOOLS:/opt/cmake:ro" \
  -v /usr/bin/git:/usr/local/bin/git:ro \
  --entrypoint bash "$BASE_IMAGE" -lc \
  'export PATH=/usr/local/bin:/usr/local/cuda/bin:$PATH CUDAToolkit_ROOT=/usr/local/cuda; /opt/cmake/data/bin/cmake --build "$1/cmake" --target libdsv41_cuda --parallel 4; nm -D "$1/cmake/libdsv41_cuda.so" | grep dsv41_engram_ssd_version' _ "$OUT"

phase image-build
cp "$OUT/cmake/libdsv41_cuda.so" "$CONTEXT/libdsv41_cuda.so"
test -r "$CONTAINERFILE" || fail "containerfile missing: $CONTAINERFILE"
timeout --foreground 1800 docker build --pull=false --tag "$IMAGE" \
  --label local.deepseek.checkpoint=official-1017-unchanged \
  --label local.deepseek.pp=5 \
  --label local.deepseek.pp-layer-partition=5,5,5,4,21 \
  --label local.deepseek.cpu-moe=layers-36-39 \
  --label local.deepseek.host-memory-reserve-gib=48 \
  --label local.deepseek.parent-image="$BASE_IMAGE" \
  --label local.deepseek.dspark=disabled \
  --label local.deepseek.engram-ssd-bridge=rebuilt \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --file "$CONTAINERFILE" "$CONTEXT"

phase image-validation
docker run --rm --network none --gpus device=0 --entrypoint python3 "$IMAGE" -c 'import torch; torch.cuda.init(); import ctypes,pathlib,vllm; from vllm.models.deepseek_v4_1.hybrid import apply_hybrid_defaults; lib=ctypes.CDLL(str(pathlib.Path(vllm.__file__).parent/"libdsv41_cuda.so")); assert lib.dsv41_engram_ssd_version()==1; print("GPU hybrid import and SSD bridge OK",torch.cuda.device_count())'
docker image inspect "$IMAGE" --format '{{.Id}} {{json .Config.Labels}}'
phase BUILD_DONE
echo "[NATIVE-BUILD-DONE] image=$IMAGE cpu_layers=36-39 pp=5,5,5,4,21"
