#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/run/media/xpz/models/deploy-llm/isolated-aiultrayang-pp5-ssd-async-20261102
SOURCE_ROOT=/run/media/xpz/models/deploy-llm/isolated-aiultrayang-pp5-ssd-async-20261031/src
PATCH=/run/media/xpz/models/deploy-llm/isolated-pp5-engram-mmap-20261031/performance-hotspots-20261101/r8-final-build/lazy-receive.patch
BASE_IMAGE=local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-20261031
IMAGE=local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-lazy-20261102
START=$(date +%s)
mkdir -p "$ROOT/reports" "$ROOT/context"
LOG="$ROOT/reports/lazy-overlay-build.log"
PHASE="$ROOT/reports/lazy-overlay-build.phase"
exec > >(tee -a "$LOG") 2>&1
phase(){ printf '%s\n' "$1" > "$PHASE"; }
heartbeat(){ while sleep 20; do echo "[HB $(( $(date +%s)-START ))s] phase=$(cat "$PHASE" 2>/dev/null || echo unknown)"; done; }
heartbeat & HB=$!
trap 'kill "$HB" 2>/dev/null || true; wait "$HB" 2>/dev/null || true' EXIT
fail(){ phase BUILD_FAIL; echo "[HB] phase=BUILD_FAIL reason=$*"; exit 1; }

phase preflight
docker image inspect "$BASE_IMAGE" >/dev/null || fail "O_DIRECT/AIO base image missing"
[[ -s "$PATCH" && -d "$SOURCE_ROOT/.git" ]] || fail "lazy patch or pinned source checkout missing"
if ! grep -q '_pp5_defer_receive' "$SOURCE_ROOT/vllm/v1/worker/gpu_worker.py"; then
  git -C "$SOURCE_ROOT" apply --check "$PATCH" || fail "lazy receive patch does not apply to the O_DIRECT source"
  git -C "$SOURCE_ROOT" apply "$PATCH"
fi
python3 -m py_compile "$SOURCE_ROOT/vllm/v1/worker/gpu_worker.py" || fail "gpu_worker compile failed"
cp "$SOURCE_ROOT/vllm/v1/worker/gpu_worker.py" "$ROOT/context/gpu_worker.py"
cat > "$ROOT/context/Containerfile" <<'EOF'
FROM local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-20261031
COPY gpu_worker.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py
LABEL local.deepseek.pp5-performance="lazy-receive" local.deepseek.engram="aiultrayang-O_DIRECT-Linux-AIO"
EOF

phase image-build
timeout --foreground --kill-after=30 600 docker build --network host \
  --tag "$IMAGE" \
  --label local.deepseek.checkpoint=official-1017-unchanged \
  --label local.deepseek.pp=5 \
  --label local.deepseek.pp-layer-partition=8,8,8,8,8 \
  --label local.deepseek.engram=aiultrayang-O_DIRECT-Linux-AIO \
  --file "$ROOT/context/Containerfile" "$ROOT/context"

phase image-validation
docker run --rm --network none --entrypoint sh "$IMAGE" -c \
  'grep -q _pp5_defer_receive /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py && test -f /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram_ssd.py && test -f /usr/local/lib/python3.12/dist-packages/vllm/libdsv41_cuda.so'
docker image inspect "$IMAGE" --format '{{.Id}} {{json .Config.Labels}}'
phase BUILD_DONE
echo "[HB] phase=BUILD_DONE image=$IMAGE"
