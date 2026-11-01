#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${ROOT:-/srv/models/deploy-llm/isolated-cpu-hybrid-pp5-20261031}
BASE_IMAGE=${BASE_IMAGE:-local/deepseek-v41-official-cpu-hybrid-pp5:20261031-r2}
IMAGE=${IMAGE:-local/deepseek-v41-official-cpu-hybrid-pp5:20261031-r3}
CONTEXT="$ROOT/gpu-resident-context"
PKG_DIR=/usr/local/lib/python3.12/dist-packages
START=$(date +%s)
mkdir -p "$CONTEXT" "$ROOT/reports"
exec > >(tee -a "$ROOT/reports/gpu-resident-build.log") 2>&1
phase() { printf '%s\n' "$1" > "$ROOT/reports/gpu-resident-build.phase"; }
heartbeat() { while sleep 20; do echo "[GPU-RESIDENT-BUILD-HB $(( $(date +%s)-START ))s] phase=$(cat "$ROOT/reports/gpu-resident-build.phase" 2>/dev/null || echo unknown)"; done; }
heartbeat & HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null || true; wait "$HB_PID" 2>/dev/null || true' EXIT
fail() { phase BUILD_FAIL; echo "[GPU-RESIDENT-BUILD-FAIL] $*"; exit 1; }

phase preflight
docker image inspect "$BASE_IMAGE" >/dev/null || fail "base image missing: $BASE_IMAGE"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then fail "image tag exists; choose a new IMAGE tag"; fi
for file in hybrid.py hybrid_runtime.py Containerfile; do test -r "$CONTEXT/$file" || fail "build input missing: $CONTEXT/$file"; done

phase image-build
echo "[GPU-RESIDENT-BUILD-HB 0s] phase=image-build base=$BASE_IMAGE output=$IMAGE cpu_layers=39-39 pp=8,8,8,8,8"
timeout --foreground 1800 docker build --pull=false --tag "$IMAGE" \
  --label local.deepseek.checkpoint=official-1017-unchanged \
  --label local.deepseek.pp=5 \
  --label local.deepseek.pp-layer-partition=8,8,8,8,8 \
  --label local.deepseek.cpu-moe=single-layer-39 \
  --label local.deepseek.host-memory-reserve-gib=48 \
  --label local.deepseek.parent-image="$BASE_IMAGE" \
  --label local.deepseek.dspark=disabled \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg PKG_DIR="$PKG_DIR" \
  --file "$CONTEXT/Containerfile" "$CONTEXT"

phase image-validation
docker run --rm --network none --gpus device=0 --entrypoint python3 "$IMAGE" -c 'import torch; torch.cuda.init(); from vllm.models.deepseek_v4_1.hybrid import apply_hybrid_defaults, cpu_expert_layer_count; from vllm.models.deepseek_v4_1.cpu_moe import CPUMoEConfig; from vllm.models.deepseek_v4_1.hybrid_runtime import HybridExecutorResources; print("GPU hybrid import OK", torch.cuda.device_count(), "single-layer helper OK")'
docker image inspect "$IMAGE" --format '{{.Id}} {{json .Config.Labels}}'
phase BUILD_DONE
echo "[GPU-RESIDENT-BUILD-DONE] image=$IMAGE cpu_layers=39-39 pp=8,8,8,8,8"
