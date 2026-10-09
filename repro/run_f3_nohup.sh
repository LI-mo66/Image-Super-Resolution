#!/usr/bin/env bash
# Run under nohup. Shutdown only after successful, validated F3 collection.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
F3_PYTHON="${F3_PYTHON:-python}"
F3_CAPTURE_DIR="$TASK_ROOT/experiment/nohup_launcher"
mkdir -p "$F3_CAPTURE_DIR"
F3_CAPTURE="$F3_CAPTURE_DIR/F3_$(date +%Y%m%d_%H%M%S)_$$.log"
"$F3_PYTHON" -u repro/run_f3_screen_server.py "$@" 2>&1 | tee "$F3_CAPTURE"
# A failing Python/pipeline exits above; never schedule shutdown on failure.
F3_GROUP="$(sed -n 's/^F3 GROUP: //p' "$F3_CAPTURE" | tail -n 1)"
if [[ -z "$F3_GROUP" ]]; then
    echo 'Missing F3 group path: shutdown refused' >&2
    exit 1
fi
"$F3_PYTHON" -u repro/shutdown_completed_run.py --group "$F3_GROUP" --invoke
