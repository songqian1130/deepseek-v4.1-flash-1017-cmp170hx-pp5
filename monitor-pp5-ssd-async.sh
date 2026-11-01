#!/usr/bin/env bash
# Stream isolated PP5 DSpark status until smoke report, container exit, or deadline.
set -u

NAME=deepseek-v41-aiultrayang-pp5-ssd-dspark-20261031
ROOT=/srv/models/deploy-llm/isolated-aiultrayang-pp5-ssd-async-20261031
BASE=http://127.0.0.1:18020
REPORT="$ROOT/reports/dspark-smoke.json"
START=$(date +%s)
DEADLINE=7200

echo "[MONITOR 0s] phase=start container=$NAME deadline_s=$DEADLINE"
while true; do
  NOW=$(date +%s)
  EL=$((NOW - START))

  if [[ -s "$REPORT" ]]; then
    RESULT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("ok"))' "$REPORT" 2>/dev/null || echo invalid)
    PASSED=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("cases_passed"))' "$REPORT" 2>/dev/null || echo unknown)
    echo "[MONITOR ${EL}s] phase=smoke-result result=$RESULT passed=$PASSED report=$REPORT"
    if [[ "$RESULT" == True || "$RESULT" == true ]]; then
      exit 0
    fi
    exit 1
  fi

  STATE=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo missing)
  if [[ "$STATE" != running ]]; then
    echo "[MONITOR ${EL}s] phase=container-exited status=$STATE"
    docker inspect "$NAME" --format 'exit={{.State.ExitCode}} error={{.State.Error}} oom={{.State.OOMKilled}}' 2>&1 || true
    docker logs --tail 50 "$NAME" 2>&1 || true
    exit 1
  fi

  LOG_PATH=$(docker inspect -f '{{.LogPath}}' "$NAME" 2>/dev/null || true)
  LOG_BYTES=$(stat -c %s "$LOG_PATH" 2>/dev/null || echo 0)
  if curl -fsS --max-time 3 "$BASE/health" >/dev/null 2>&1; then
    MODELS=$(curl -fsS --max-time 5 "$BASE/v1/models" 2>/dev/null || true)
    if [[ "$MODELS" == *deepseek-v4.1-flash-1017* ]]; then
      PHASE=health-ready
    else
      PHASE=api-starting
    fi
  else
    PHASE=loading
  fi

  LAST=$(docker logs --tail 1 "$NAME" 2>&1 | tr '\r' '\n' | tail -1 | cut -c1-180)
  echo "[MONITOR ${EL}s] phase=$PHASE container=$STATE log_bytes=$LOG_BYTES last=$LAST"
  if (( EL >= DEADLINE )); then
    echo "[MONITOR ${EL}s] phase=deadline-reached"
    exit 124
  fi
  sleep 20
done
