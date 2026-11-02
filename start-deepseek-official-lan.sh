#!/usr/bin/env bash
set -Eeuo pipefail
# Compatibility alias: always use the same current production entry point.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$HERE/start-deepseek-official.sh" "$@"
