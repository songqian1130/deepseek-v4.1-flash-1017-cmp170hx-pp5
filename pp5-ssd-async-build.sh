#!/usr/bin/env bash
# Build an isolated PP5 image with aiultrayang's C++ O_DIRECT + Linux AIO Engram backend.
set -Eeuo pipefail

ROOT=/srv/models/deploy-llm/isolated-aiultrayang-pp5-ssd-async-20261031
SOURCE_URL=https://github.com/344303947/dsv41-flash-pp5-170hx.git
SOURCE_COMMIT=d63af5a472dc76b12d7a73d50a5af142844c15d1
REFERENCE_COMMIT=29b03af5eedfac9af12dfa14e8cfa7e2eafc852a
REFERENCE_RAW=https://raw.githubusercontent.com/aiultrayang/deepseek-v41-flash-cmp170hx/${REFERENCE_COMMIT}/pp6-ssd
BASE_IMAGE=lazymio/vllm-backport@sha256:bfe237ee0886d9b4b2357b84bfbe09805678cc80b2182dd2a87cefb89497342d
IMAGE=${IMAGE:-local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-dspark-head-uva-20261031}

mkdir -p "$ROOT" "$ROOT/reports"
LOG="$ROOT/reports/build.log"
PHASE_FILE="$ROOT/reports/build.phase"
phase() { printf '%s\n' "$1" > "$PHASE_FILE"; }
phase preflight
exec > >(tee -a "$LOG") 2>&1
START=$(date +%s)
heartbeat() {
  while sleep 20; do
    EL=$(( $(date +%s) - START ))
    CURRENT_PHASE=$(cat "$PHASE_FILE" 2>/dev/null || echo unknown)
    echo "[HB ${EL}s] phase=${CURRENT_PHASE}"
  done
}
heartbeat & HB_PID=$!
finish() { kill "$HB_PID" 2>/dev/null || true; wait "$HB_PID" 2>/dev/null || true; }
trap finish EXIT
fail() { phase BUILD_FAIL; echo "[HB] phase=BUILD_FAIL reason=$*"; exit 1; }

echo "[HB 0s] phase=preflight root=$ROOT"
docker image inspect "$BASE_IMAGE" >/dev/null || fail "pinned base image missing locally"
for path in \
  /srv/models/DeepSeek-V4.1-Flash-1017-official/config.json \
  /srv/models/DeepSeek-V4.1-Flash-1017-official/model-00047-of-00048.safetensors \
  /srv/models/DeepSeek-V4.1-Flash-1017-official/model-00048-of-00048.safetensors \
  /home/llm/.cache/deepseek-v41-official/model-00048-of-00048.safetensors; do
  test -r "$path" || fail "required read-only checkpoint path missing: $path"
done

phase source-fetch
mkdir -p "$ROOT/src"
if [[ ! -d "$ROOT/src/.git" ]]; then
  git -C "$ROOT/src" init -q
  git -C "$ROOT/src" remote add origin "$SOURCE_URL"
fi
# Reuse the verified commit already cached in the isolated workspace when GitHub is unreachable.
if ! git -C "$ROOT/src" cat-file -e "${SOURCE_COMMIT}^{commit}" 2>/dev/null; then
  git -C "$ROOT/src" fetch --depth 1 origin "$SOURCE_COMMIT"
fi
git -C "$ROOT/src" checkout -q --force --detach "$SOURCE_COMMIT"
git -C "$ROOT/src" clean -qfd
ACTUAL=$(git -C "$ROOT/src" rev-parse HEAD)
[[ "$ACTUAL" == "$SOURCE_COMMIT" ]] || fail "unexpected PP5 source commit $ACTUAL"

phase reference-download
mkdir -p "$ROOT/reference" "$ROOT/src/csrc/cpu/dsv41"
PATCH_COMMUNICATOR="$ROOT/reference/0001-prime-pp-communicators-before-model-load.patch"
PATCH_ENGRAM="$ROOT/reference/engram-ssd-port.patch"
PATCH_DSPARK="$ROOT/reference/dspark-construction-offload.patch"
CONTAINERFILE="$ROOT/Containerfile"
if [[ ! -s "$PATCH_COMMUNICATOR" ]]; then curl -fL --connect-timeout 15 --max-time 90 "$REFERENCE_RAW/patches/0001-prime-pp-communicators-before-model-load.patch" -o "$PATCH_COMMUNICATOR"; fi
if [[ ! -s "$PATCH_ENGRAM" ]]; then curl -fL --connect-timeout 15 --max-time 90 "$REFERENCE_RAW/patches/engram-ssd-port.patch" -o "$PATCH_ENGRAM"; fi
if [[ ! -s "$CONTAINERFILE" ]]; then curl -fL --connect-timeout 15 --max-time 90 "$REFERENCE_RAW/build/Containerfile" -o "$CONTAINERFILE"; fi
printf '%s  %s\n' \
  16d8b755470052a7e939a10c97e6877396b55d06b24b548c166c7e5974dd5af7 "$PATCH_COMMUNICATOR" \
  5ec439e0d300cee47fa5ba9ee99c393d71161cae95131298ada31b0a51c658c1 "$PATCH_ENGRAM" \
  5af7be1827951c7d883dcc7636be7c2c82a44092b016f33b1468856799649689 "$CONTAINERFILE" | sha256sum -c -

if [[ ! -s "$PATCH_DSPARK" ]]; then
  fail "missing durable DSpark construction patch: $PATCH_DSPARK"
fi

echo "[HB] phase=patch-check"
git -C "$ROOT/src" apply --check "$PATCH_COMMUNICATOR" || fail "PP communicator patch does not apply"
git -C "$ROOT/src" apply --check "$PATCH_ENGRAM" || fail "Engram SSD patch does not apply"
git -C "$ROOT/src" apply --check "$PATCH_DSPARK" || fail "DSpark construction patch does not apply"
git -C "$ROOT/src" apply "$PATCH_COMMUNICATOR"
git -C "$ROOT/src" apply "$PATCH_ENGRAM"
git -C "$ROOT/src" apply "$PATCH_DSPARK"
python3 -m py_compile \
  "$ROOT/src/vllm/models/deepseek_v4_1/common/engram.py" \
  "$ROOT/src/vllm/models/deepseek_v4_1/common/engram_ssd.py" \
  "$ROOT/src/vllm/models/deepseek_v4_1/host_memory.py" \
  "$ROOT/src/vllm/config/engram.py" "$ROOT/src/vllm/config/vllm.py" \
  "$ROOT/src/vllm/v1/worker/gpu_worker.py"
grep -q '_pp_communicators_primed' "$ROOT/src/vllm/v1/worker/gpu_worker.py"
test -s "$ROOT/src/csrc/cpu/dsv41/engram_ssd.cpp"

phase build-context
rm -rf "$ROOT/context"
mkdir -p "$ROOT/context/native"
cp -a "$ROOT/src/vllm" "$ROOT/context/vllm"
cp "$ROOT/src/csrc/cpu/dsv41/engram_ssd.cpp" "$ROOT/context/native/engram_ssd.cpp"
PKG_DIR=$(docker run --rm --network none --entrypoint python3 "$BASE_IMAGE" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
test -n "$PKG_DIR" || fail "could not resolve Python package path"

echo "[HB] phase=image-build base=$BASE_IMAGE output=$IMAGE"
phase image-build
timeout --foreground 1800 docker build --pull=false \
  --tag "$IMAGE" \
  --label local.deepseek.checkpoint=official-1017-unchanged \
  --label local.deepseek.pp=5 \
  --label local.deepseek.pp-layer-partition=8,8,8,8,8 \
  --label local.deepseek.engram=aiultrayang-O_DIRECT-Linux-AIO \
  --label local.deepseek.reference-commit="$REFERENCE_COMMIT" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg PKG_DIR="$PKG_DIR" \
  --file "$ROOT/Containerfile" "$ROOT/context"

phase image-validation
docker run --rm --network none --entrypoint python3 "$IMAGE" -c \
  'import ctypes,pathlib,vllm; p=pathlib.Path(vllm.__file__).parent/"libdsv41_cuda.so"; lib=ctypes.CDLL(str(p)); lib.dsv41_engram_ssd_version.restype=ctypes.c_int; print("vllm",vllm.__version__); print("native",p,"version",lib.dsv41_engram_ssd_version()); assert lib.dsv41_engram_ssd_version()==1'
docker image inspect "$IMAGE" --format '{{.Id}} {{json .Config.Labels}}'
echo "[HB] phase=BUILD_DONE image=$IMAGE source=$SOURCE_COMMIT"
