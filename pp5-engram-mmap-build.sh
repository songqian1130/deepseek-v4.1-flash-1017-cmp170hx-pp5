#!/usr/bin/env bash
set -Eeuo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=/srv/models/deploy-llm/isolated-pp5-engram-mmap-20261031
BASE_IMAGE=local/deepseek-v41-official-pp5:20261031
IMAGE=local/deepseek-v41-official-pp5:alex-engram-mmap-20261031-r1
SOURCE_COMMIT=b3d5e18081285f75eb55e1e7a1c45e209d612e35
LOG="$ROOT/reports/build.log"
PHASE_FILE="$ROOT/reports/build.phase"

mkdir -p "$ROOT/reports"
exec > >(tee -a "$LOG") 2>&1
START=$(date +%s)
phase() { printf '%s\n' "$1" > "$PHASE_FILE"; }
heartbeat() {
  while sleep 20; do
    EL=$(( $(date +%s) - START ))
    echo "[HB ${EL}s] phase=$(cat "$PHASE_FILE" 2>/dev/null || echo unknown)"
  done
}
heartbeat & HB_PID=$!
finish() { kill "$HB_PID" 2>/dev/null || true; wait "$HB_PID" 2>/dev/null || true; }
trap finish EXIT
fail() { phase BUILD_FAIL; echo "[HB] phase=BUILD_FAIL reason=$*"; exit 1; }

phase preflight
echo "[HB 0s] phase=preflight base=$BASE_IMAGE image=$IMAGE"
docker image inspect "$BASE_IMAGE" >/dev/null || fail "pinned PP5 base image is missing"
ACTUAL_SOURCE=$(docker image inspect -f '{{index .Config.Labels "local.deepseek.pp5-source"}}' "$BASE_IMAGE")
[[ "$ACTUAL_SOURCE" == "$SOURCE_COMMIT" ]] || fail "base image source commit mismatch: $ACTUAL_SOURCE"
[[ -s "$HERE/engram_mmap.py" ]] || fail "missing safetensors mmap helper"
[[ -s "$HERE/disk_engram_mmap.py" ]] || fail "missing DiskEngramRows mmap adapter"
[[ -s "$HERE/install-pp5-engram-mmap.py" ]] || fail "missing image installer"
[[ -s "$HERE/verify-real-engram-mmap.py" ]] || fail "missing real-checkpoint verifier"
docker image inspect "$IMAGE" >/dev/null 2>&1 && fail "target image tag already exists: $IMAGE"

phase image-build
timeout --foreground 1800 docker build --pull=false --network host \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --file "$HERE/pp5-engram-mmap.Containerfile" \
  --tag "$IMAGE" "$HERE"

ACTUAL_TAG=$(docker image inspect -f '{{index .Config.Labels "local.deepseek.engram"}}' "$IMAGE")
[[ "$ACTUAL_TAG" == alexbrandes-mmap-fp8-row-gather ]] || fail "new image label mismatch: $ACTUAL_TAG"
ACTUAL_SOURCE=$(docker image inspect -f '{{index .Config.Labels "local.deepseek.pp5-source"}}' "$IMAGE")
[[ "$ACTUAL_SOURCE" == "$SOURCE_COMMIT" ]] || fail "new image PP5 source label mismatch: $ACTUAL_SOURCE"
phase BUILD_DONE
echo "[HB] phase=BUILD_DONE image=$IMAGE source=$SOURCE_COMMIT"
