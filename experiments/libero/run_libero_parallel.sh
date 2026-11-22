#!/usr/bin/env bash
# Compatibility entry for existing Hydra/cluster commands.
set -euo pipefail
if [[ -n "${WORKER_ENV_SOURCE:-}" ]]; then
  source "$WORKER_ENV_SOURCE"
fi
exec python "$(dirname "$0")/run_libero_workers.py" "$@"
