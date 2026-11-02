#!/usr/bin/env bash
set -Eeuo pipefail

# No-argument entry point for the verified PP5 service.
# The API is exposed on the LAN at http://192.168.3.44:9010/v1.
BASE=/run/media/xpz/models/deploy-llm
CONTAINER=deepseek-v41-pp5-aio-9010-20261102
MODEL=deepseek-v4.1-flash-1017
URL=http://127.0.0.1:9010

if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  state=$(docker inspect -f '{{.State.Status}}' "$CONTAINER")
  if [[ "$state" != running ]]; then
    timeout --foreground 60 docker start "$CONTAINER" >/dev/null
  fi
else
  echo "Verified container is missing: $CONTAINER. See aio-stability-20261102 recovery files." >&2
  exit 1
fi

deadline=$(( $(date +%s) + 1200 ))
while (( $(date +%s) < deadline )); do
  if curl --noproxy '*' --silent --fail --max-time 5 "$URL/health" >/dev/null \
      && curl --noproxy '*' --silent --fail --max-time 5 "$URL/v1/models" | grep -q "$MODEL"; then
    echo "DeepSeek PP5 is ready."
    echo "API: http://192.168.3.44:9010/v1"
    echo "Recording gateway: http://192.168.3.44:8081/v1"
    echo "Model: $MODEL"
    exit 0
  fi
  echo "[HB $(date +%s)] phase=wait-deepseek-pp5 container=$CONTAINER"
  sleep 15
done
echo "DeepSeek PP5 did not become ready within 1200 seconds." >&2
echo "Logs: docker logs $CONTAINER; reports: $BASE/aio-stability-20261102" >&2
exit 1
