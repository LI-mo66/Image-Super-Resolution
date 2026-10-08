#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python -u "$SCRIPT_DIR/run_v1_combos_screen_server.py" "$@"
