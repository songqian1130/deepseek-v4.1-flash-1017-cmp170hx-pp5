#!/usr/bin/env bash
set -Eeuo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-/srv/models/deploy-llm/isolated-pp5-engram-mmap-20261031}
BASE_IMAGE=${BASE_IMAGE:-local/deepseek-v41-official-pp5:alex-engram-mmap-20261031-r1}
IMAGE=${IMAGE:-local/deepseek-v41-official-pp5:alex-engram-mmap-20261031-r2-dspark}
SOURCE_COMMIT=b3d5e18081285f75eb55e1e7a1c45e209d612e35
LOG="$ROOT/reports/dspark-build.log"
PHASE_FILE="$ROOT/reports/dspark-build.phase"
CONTEXT="$ROOT/dspark-build-context"

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
docker image inspect "$BASE_IMAGE" >/dev/null || fail "verified mmap r1 base is missing"
ACTUAL_SOURCE=$(docker image inspect -f '{{index .Config.Labels "local.deepseek.pp5-source"}}' "$BASE_IMAGE")
[[ "$ACTUAL_SOURCE" == "$SOURCE_COMMIT" ]] || fail "base image source commit mismatch: $ACTUAL_SOURCE"
[[ $(docker image inspect -f '{{index .Config.Labels "local.deepseek.engram"}}' "$BASE_IMAGE") == alexbrandes-mmap-fp8-row-gather ]] || fail "base image does not contain the verified Engram mmap"
[[ -s "$HERE/dspark-construction-offload.patch" ]] || fail "DSpark construction patch is missing"
docker image inspect "$IMAGE" >/dev/null 2>&1 && fail "target image tag already exists: $IMAGE"

phase image-build
mkdir -p "$CONTEXT"
cp "$HERE/pp5-engram-mmap-dspark.Containerfile" "$HERE/dspark-construction-offload.patch" "$CONTEXT/"
timeout --foreground 900 docker build --pull=false --network host \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --file "$CONTEXT/pp5-engram-mmap-dspark.Containerfile" \
  --tag "$IMAGE" "$CONTEXT"

[[ $(docker image inspect -f '{{index .Config.Labels "local.deepseek.engram"}}' "$IMAGE") == alexbrandes-mmap-fp8-row-gather ]] || fail "Engram mmap label was not preserved"
[[ $(docker image inspect -f '{{index .Config.Labels "local.deepseek.dspark"}}' "$IMAGE") == independent-offloader-uva-head ]] || fail "DSpark patch label mismatch"
[[ $(docker image inspect -f '{{index .Config.Labels "local.deepseek.pp5-source"}}' "$IMAGE") == "$SOURCE_COMMIT" ]] || fail "PP5 source label mismatch"
phase BUILD_DONE
echo "[HB] phase=BUILD_DONE image=$IMAGE source=$SOURCE_COMMIT"
