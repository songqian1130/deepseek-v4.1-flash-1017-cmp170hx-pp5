#!/usr/bin/env bash
set -Eeuo pipefail
if [[ ${AIO_STABILITY_GUARD:-0} != 1 ]]; then
  exec timeout --foreground --kill-after=900 3600 env AIO_STABILITY_GUARD=1 bash "$0"
fi
ROOT=/run/media/xpz/models/deploy-llm/aio-stability-20261102
BASE=deepseek-v41-pp5-trap31-p2p-20261102
CAND=deepseek-v41-aiultrayang-pp5-ssd-matched-20261102
mkdir -p "$ROOT/reports"
exec >>"$ROOT/run.log" 2>&1
START=$(date +%s)
phase(){ echo "$1" > "$ROOT/phase"; echo "[HB $(( $(date +%s)-START ))s] phase=$1"; }
heartbeat(){ while sleep 15; do echo "[HB $(( $(date +%s)-START ))s] phase=$(cat "$ROOT/phase")"; done; }
heartbeat & HB=$!
ready(){
  local port=$1 name=$2 limit=$3 deadline=$(( $(date +%s)+$3 ))
  while (( $(date +%s) < deadline )); do
    [[ $(docker inspect -f '{{.State.Status}}' "$name") == running ]] || return 1
    if curl -fsS --max-time 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then return 0; fi
    sleep 5
  done
  return 1
}
idle(){
  curl -fsS --max-time 5 "http://127.0.0.1:$1/metrics" |
    python3 -c 'import sys; lines=[l for l in sys.stdin if l.startswith(("vllm:num_requests_running{","vllm:num_requests_waiting{"))]; assert len(lines)==2 and all(float(l.rsplit(None,1)[1])==0 for l in lines), lines'
}
restore(){
  local rc=$?
  trap - EXIT INT TERM
  set +e
  phase restore
  timeout --kill-after=15 150 docker stop --time 120 "$CAND"
  if [[ $(docker inspect -f '{{.State.Status}}' "$BASE") != running ]]; then timeout 60 docker start "$BASE"; fi
  if ready 9010 "$BASE" 850 && idle 9010; then
    curl -fsS --max-time 5 -o /dev/null -w 'health=%{http_code}\n' http://127.0.0.1:9010/health
    docker inspect -f '{{.Name}} state={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' "$BASE" "$CAND"
    echo "[HB] phase=RESTORE_DONE rc=$rc"
  else
    rc=1
    echo '[HB] phase=RESTORE_FAILED'
  fi
  echo "$rc" > "$ROOT/exit-code"
  kill "$HB" 2>/dev/null; wait "$HB" 2>/dev/null
  exit "$rc"
}
trap restore EXIT
trap 'exit 130' INT TERM
phase preflight
[[ $(docker inspect -f '{{.State.Status}}' "$BASE") == running ]]
[[ $(docker inspect -f '{{.State.Status}}' "$CAND") == exited ]]
idle 9010
docker inspect "$BASE" "$CAND" > "$ROOT/reports/containers.json"
bench(){ phase "$1"; timeout --foreground --kill-after=15 650 python3 "$ROOT/benchmark-aio-stability.py" --base "http://127.0.0.1:$2/v1" --container "$3" --out "$ROOT/reports/$1.json"; idle "$2"; }
bench baseline-a1 9010 "$BASE"
phase candidate-start
timeout --kill-after=15 150 docker stop --time 120 "$BASE"
timeout 60 docker start "$CAND"
ready 18023 "$CAND" 1050
bench candidate-b 18023 "$CAND"
phase baseline-restart
timeout --kill-after=15 150 docker stop --time 120 "$CAND"
docker logs --since 1h "$CAND" > "$ROOT/reports/candidate.log" 2>&1
timeout 60 docker start "$BASE"
ready 9010 "$BASE" 850
bench baseline-a2 9010 "$BASE"
phase TEST_DONE
