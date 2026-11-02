#!/usr/bin/env bash
set -Eeuo pipefail
if [[ ${AIO_PROMOTE_GUARD:-0} != 1 ]]; then
  exec timeout --foreground --kill-after=950 1800 env AIO_PROMOTE_GUARD=1 bash "$0"
fi
ROOT=/run/media/xpz/models/deploy-llm/aio-stability-20261102
BASE=deepseek-v41-pp5-trap31-p2p-20261102
DEST=deepseek-v41-pp5-aio-9010-20261102
ENTRY=/run/media/xpz/models/deploy-llm/start-deepseek-official.sh
START=$(date +%s)
exec >>"$ROOT/promotion.log" 2>&1
phase(){ echo "$1" > "$ROOT/promotion.phase"; echo "[HB $(( $(date +%s)-START ))s] phase=$1"; }
heartbeat(){ while sleep 15; do echo "[HB $(( $(date +%s)-START ))s] phase=$(cat "$ROOT/promotion.phase")"; done; }
heartbeat & HB=$!
SWITCHED=0
SUCCESS=0
ready(){
  local name=$1 limit=$2 deadline=$(( $(date +%s)+$2 ))
  while (( $(date +%s) < deadline )); do
    [[ $(docker inspect -f '{{.State.Status}}' "$name") == running ]] || return 1
    if curl -fsS --max-time 3 http://127.0.0.1:9010/health >/dev/null 2>&1 && curl -fsS --max-time 3 http://127.0.0.1:9010/v1/models >/dev/null 2>&1; then return 0; fi
    sleep 5
  done
  return 1
}
cleanup(){
  local rc=$?
  trap - EXIT INT TERM
  set +e
  if [[ $SUCCESS != 1 && $SWITCHED == 1 ]]; then
    phase rollback
    timeout --kill-after=15 150 docker stop --time 120 "$DEST"
    cp "$ROOT/reports/start-deepseek-official.before.sh" "$ENTRY"
    timeout 60 docker start "$BASE"
    if ready "$BASE" 850; then echo '[HB] phase=ROLLBACK_READY'; else echo '[HB] phase=ROLLBACK_FAILED'; fi
  fi
  echo "$rc" > "$ROOT/promotion-exit-code"
  kill "$HB" 2>/dev/null; wait "$HB" 2>/dev/null
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
phase preflight
[[ $(cat "$ROOT/exit-code") == 0 ]]
python3 "$ROOT/analyze-aio-stability.py" "$ROOT/reports" > "$ROOT/reports/analysis-summary.txt"
python3 - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1])
a=json.loads((r/'reports/analysis.json').read_text())
assert all(a[n]['done'] for n in ('baseline-a1','candidate-b','baseline-a2'))
candidate=a['candidate-b']['phases']
for n in ('baseline-a1','baseline-a2'):
    for phase in ('repeated','varied'):
        base=a[n]['phases'][phase]; cand=candidate[phase]
        assert cand['median'] > base['median']*1.02,(n,phase,'median')
        assert cand['e2e_weighted'] > base['e2e_weighted']*1.02,(n,phase,'e2e')
        assert cand['cpu_ms_per_token']['usage_usec'] < base['cpu_ms_per_token']['usage_usec']*.9,(n,phase,'cpu')
for phase in ('repeated','varied','natural'):
    assert not candidate[phase]['rounds_gap_over1s']
    assert not candidate[phase]['memory_events_delta']['oom']
print('PROMOTION_GATE_PASS: both baselines beaten on fixed-length throughput and CPU; no >1s measured SSE gaps')
PY
curl -fsS --max-time 5 http://127.0.0.1:9010/metrics | python3 -c 'import sys; a=[l for l in sys.stdin if l.startswith(("vllm:num_requests_running{","vllm:num_requests_waiting{"))]; assert len(a)==2 and all(float(l.rsplit(None,1)[1])==0 for l in a)'
cp "$ENTRY" "$ROOT/reports/start-deepseek-official.before.sh"
python3 "$ROOT/clone-aio-9010.py"
phase switch
SWITCHED=1
timeout --kill-after=15 150 docker stop --time 120 "$BASE"
timeout 60 docker start "$DEST"
ready "$DEST" 1050
phase smoke
timeout --foreground --kill-after=15 180 python3 "$ROOT/verify-pp5-ssd-async.py" --base http://127.0.0.1:9010/v1 --model deepseek-v4.1-flash-1017 --out "$ROOT/reports/promoted-smoke.json"
phase acceptance
timeout --foreground --kill-after=15 240 python3 "$ROOT/benchmark-aio-stability.py" --base http://127.0.0.1:9010/v1 --container "$DEST" --acceptance --out "$ROOT/reports/promoted-9010.json"
phase update-entrypoint
python3 - "$ENTRY" "$ROOT" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); text=p.read_text()
assert text.count('CONTAINER=deepseek-v41-pp5-trap31-p2p-20261102')==1
text=text.replace('CONTAINER=deepseek-v41-pp5-trap31-p2p-20261102','CONTAINER=deepseek-v41-pp5-aio-9010-20261102')
text=text.replace('See p2p-trap31-20261102 recovery files.','See aio-stability-20261102 recovery files.')
p.write_text(text)
PY
bash -n "$ENTRY"
timeout --foreground 30 bash "$ENTRY"
curl -fsS --max-time 5 http://127.0.0.1:9010/metrics > "$ROOT/reports/promoted-metrics.txt"
python3 - "$ROOT/reports/promoted-metrics.txt" <<'PY'
from pathlib import Path
import sys
lines=[l for l in Path(sys.argv[1]).read_text().splitlines() if l.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
assert len(lines)==2 and all(float(l.rsplit(None,1)[1])==0 for l in lines),lines
print('IDLE_VERIFIED',lines)
PY
docker inspect -f '{{.Name}} state={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} image={{.Image}}' "$BASE" "$DEST"
SUCCESS=1
phase PROMOTION_DONE
