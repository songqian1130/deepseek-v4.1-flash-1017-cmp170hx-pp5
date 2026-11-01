#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${ROOT:-/srv/models/deploy-llm/isolated-cpu-hybrid-pp5-20261031}
BASE_IMAGE=${BASE_IMAGE:-local/deepseek-v41-official-cpu-hybrid-pp5:20261031-r1}
IMAGE=${IMAGE:-local/deepseek-v41-official-cpu-hybrid-pp5:20261031-r2}
CONTEXT="$ROOT/memory-guard-context"
START=$(date +%s)
mkdir -p "$CONTEXT" "$ROOT/reports"
exec > >(tee -a "$ROOT/reports/memory-build.log") 2>&1
phase() { printf '%s\n' "$1" > "$ROOT/reports/memory-build.phase"; }
heartbeat() { while sleep 20; do echo "[MEMORY-BUILD-HB $(( $(date +%s)-START ))s] phase=$(cat "$ROOT/reports/memory-build.phase" 2>/dev/null || echo unknown)"; done; }
heartbeat & HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null || true; wait "$HB_PID" 2>/dev/null || true' EXIT
fail() { phase BUILD_FAIL; echo "[MEMORY-BUILD-FAIL] $*"; exit 1; }

phase preflight
docker image inspect "$BASE_IMAGE" >/dev/null || fail "base image missing: $BASE_IMAGE"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then fail "image tag exists; choose a new IMAGE tag"; fi
cat > "$CONTEXT/Containerfile" <<'EOF'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN python3 -c 'from pathlib import Path; p=Path("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/host_memory.py"); s=p.read_text(); old="if host_available_bytes() < 64 * GiB + additional_bytes:"; new="if host_available_bytes() < 48 * GiB + additional_bytes:"; assert s.count(old)==1, "expected exactly one 64 GiB host headroom guard"; p.write_text(s.replace(old,new))' \
 && python3 -c 'from pathlib import Path; p=Path("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/host_memory.py"); s=p.read_text(); assert "48 * GiB + additional_bytes" in s and "64 * GiB + additional_bytes" not in s; print("host memory reserve: 48 GiB")'
EOF

phase image-build
echo "[MEMORY-BUILD-HB 0s] phase=image-build base=$BASE_IMAGE output=$IMAGE reserve_gib=48"
timeout --foreground 1800 docker build --pull=false --tag "$IMAGE" \
  --label local.deepseek.checkpoint=official-1017-unchanged \
  --label local.deepseek.pp=5 \
  --label local.deepseek.pp-layer-partition=5,5,5,4,21 \
  --label local.deepseek.cpu-moe=reference-buliaoyin \
  --label local.deepseek.host-memory-reserve-gib=48 \
  --label local.deepseek.parent-image="$BASE_IMAGE" \
  --label local.deepseek.dspark=disabled \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --file "$CONTEXT/Containerfile" "$CONTEXT"

phase image-validation
docker run --rm --network none --gpus device=0 --entrypoint python3 "$IMAGE" -c 'import torch; torch.cuda.init(); from pathlib import Path; p=Path("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/host_memory.py"); s=p.read_text(); assert "48 * GiB + additional_bytes" in s; import vllm.models.deepseek_v4_1.hybrid; import vllm.models.deepseek_v4_1.cpu_moe; print("GPU hybrid import OK", torch.cuda.device_count())'
docker image inspect "$IMAGE" --format '{{.Id}} {{json .Config.Labels}}'
phase BUILD_DONE
echo "[MEMORY-BUILD-DONE] image=$IMAGE reserve_gib=48"
