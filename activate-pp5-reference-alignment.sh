#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE=/srv/models/deploy-llm/isolated-pp5-engram-mmap-20261031
OLD=deepseek-v41-pp5-alex-engram-mmap-20261031-original-9010-r3
NEW=deepseek-v41-pp5-alex-engram-mmap-20261031-original-9010-r4
IMAGE=local/deepseek-v41-official-pp5:alex-engram-mmap-20261031-r4-reference-aligned
ROOT=$BASE/pp5-original-9010-r4-reference-aligned
mkdir -p "$ROOT/reports"
START=$(date +%s)
heartbeat() { while sleep 20; do echo "[HB $(( $(date +%s)-START ))s] phase=activate-r4"; done; }
heartbeat & HB=$!
trap 'kill "$HB" 2>/dev/null || true; wait "$HB" 2>/dev/null || true' EXIT
docker image inspect "$IMAGE" >/dev/null
timeout --foreground 45 docker run --rm --network host --entrypoint python3 "$IMAGE" \
  /opt/source-alignment/verify-pp5-reference-alignment.py \
  --package /usr/local/lib/python3.12/dist-packages/vllm \
  --manifest /opt/source-alignment/alignment-manifest.json \
  --reference-engram /opt/source-alignment/reference-engram.py
test "$(cat "$BASE/pp5-original-9010-r3-capture-ids/reports/run.phase")" = RUN_DONE
if docker inspect "$NEW" >/dev/null 2>&1; then echo 'R4_NAME_ALREADY_EXISTS'; exit 1; fi
# Existing tests must be finished and no user request may be active during switch.
curl -fsS --max-time 5 http://127.0.0.1:9010/metrics > "$ROOT/reports/r3-before-switch.metrics"
python3 - "$ROOT/reports/r3-before-switch.metrics" <<'PY'
import re,sys
text=open(sys.argv[1]).read()
samples=re.findall(r'^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*?\})?\s+([0-9.eE+-]+)',text,re.M)
assert samples and all(float(v)==0 for v in samples), 'Requests are active; refusing to interrupt'
PY
echo "[HB] phase=switch-stop-completed-r3 old=$OLD"
timeout --foreground 180 docker stop --time 120 "$OLD"
if timeout --foreground 10800 env ROOT="$ROOT" IMAGE="$IMAGE" NAME="$NEW" \
    CACHE_ROOT="$BASE/pp5-original-9010-r3-capture-ids/cache" \
    bash "$HERE/start_vllm_v41_pp5.sh"; then
  echo '[HB] phase=REFERENCE_ALIGNMENT_ACTIVATION_DONE'
else
  RC=$?
  echo "[HB] phase=R4_FAILED_ROLLBACK exit=$RC"
  timeout --foreground 180 docker stop --time 120 "$NEW" || true
  timeout --foreground 30 docker start "$OLD"
  echo '[HB] phase=R3_ROLLBACK_RESTARTED readiness-not-verified'
  exit "$RC"
fi
