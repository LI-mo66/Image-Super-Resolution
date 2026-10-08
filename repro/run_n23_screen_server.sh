#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
exec python "$SCRIPT_DIR/run_n23_screen_server.py" "$@"
