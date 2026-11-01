#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${ROOT:-/srv/models/deploy-llm/isolated-cpu-hybrid-pp5-20261031}
REF=${REF:-/srv/models/deploy-llm/reference-buliaoyin-cpu-hybrid-20261031/src}
NATIVE=${NATIVE:-/srv/models/deploy-llm/isolated-cpu-moe-pp5-20261031/container-build/work/build}
BASE_IMAGE=${BASE_IMAGE:-local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-dspark-head-uva-20261031}
IMAGE=${IMAGE:-local/deepseek-v41-official-cpu-hybrid-pp5:20261031-r1}
PKG_DIR=/usr/local/lib/python3.12/dist-packages
START=$(date +%s)
mkdir -p "$ROOT/reports"
exec > >(tee -a "$ROOT/reports/build.log") 2>&1
phase() { printf '%s\n' "$1" > "$ROOT/reports/build.phase"; }
heartbeat() { while sleep 20; do echo "[CPU-HYBRID-BUILD-HB $(( $(date +%s)-START ))s] phase=$(cat "$ROOT/reports/build.phase" 2>/dev/null || echo unknown)"; done; }
heartbeat & HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null || true; wait "$HB_PID" 2>/dev/null || true' EXIT
fail() { phase BUILD_FAIL; echo "[CPU-HYBRID-BUILD-FAIL] $*"; exit 1; }
phase preflight
echo "[CPU-HYBRID-BUILD-HB 0s] phase=preflight image=$IMAGE"
docker image inspect "$BASE_IMAGE" >/dev/null || fail "base image missing: $BASE_IMAGE"
for path in "$REF/vllm/models/deepseek_v4_1/hybrid_runtime.py" "$REF/vllm/models/deepseek_v4_1/cpu_moe.py" "$REF/vllm/model_executor/layers/fused_moe/experts/cpu_mxfp4.py" "$NATIVE/libdsv41_ik.so" "$NATIVE/libdsv41_cuda.so"; do test -r "$path" || fail "required reference artifact missing: $path"; done
if docker image inspect "$IMAGE" >/dev/null 2>&1; then fail "image tag exists; choose a new IMAGE tag"; fi
phase build-context
CONTEXT="$ROOT/context"
mkdir -p "$CONTEXT/vllm" "$CONTEXT/native"
cp -a "$REF/vllm/." "$CONTEXT/vllm/"
cp "$NATIVE/libdsv41_ik.so" "$NATIVE/libdsv41_cuda.so" "$CONTEXT/native/"
cat > "$CONTEXT/Containerfile" <<'EOF'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG PKG_DIR=/usr/local/lib/python3.12/dist-packages
COPY vllm/ ${PKG_DIR}/vllm/
COPY native/libdsv41_ik.so native/libdsv41_cuda.so ${PKG_DIR}/vllm/
RUN python3 -m compileall -q ${PKG_DIR}/vllm/models/deepseek_v4_1 ${PKG_DIR}/vllm/model_executor/layers/fused_moe/experts && python3 -c 'import ctypes,pathlib,vllm; p=pathlib.Path(vllm.__file__).parent; ctypes.CDLL(str(p/"libdsv41_ik.so")); ctypes.CDLL(str(p/"libdsv41_cuda.so")); print("vllm",vllm.__version__)'
EOF
phase image-build
echo "[CPU-HYBRID-BUILD] phase=image-build base=$BASE_IMAGE output=$IMAGE"
timeout --foreground 1800 docker build --pull=false --tag "$IMAGE" --label local.deepseek.checkpoint=official-1017-unchanged --label local.deepseek.pp=5 --label local.deepseek.pp-layer-partition=5,5,5,4,21 --label local.deepseek.cpu-moe=reference-buliaoyin --label local.deepseek.dspark=disabled --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg PKG_DIR="$PKG_DIR" --file "$CONTEXT/Containerfile" "$CONTEXT"
phase image-validation
docker run --rm --network none --entrypoint python3 "$IMAGE" -c 'import ctypes,pathlib,vllm; from vllm.models.deepseek_v4_1.hybrid import apply_hybrid_defaults; p=pathlib.Path(vllm.__file__).parent; ctypes.CDLL(str(p/"libdsv41_ik.so")); ctypes.CDLL(str(p/"libdsv41_cuda.so")); print("image",vllm.__version__,"hybrid imports OK")'
docker image inspect "$IMAGE" --format '{{.Id}} {{json .Config.Labels}}'
phase BUILD_DONE
echo "[CPU-HYBRID-BUILD-DONE] image=$IMAGE"
