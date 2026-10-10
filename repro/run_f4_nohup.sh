#!/usr/bin/env bash
# Run under nohup. Shutdown only after successful, validated F4 collection.
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
F4_PYTHON="${F4_PYTHON:-python}"
F4_CAPTURE_DIR="$TASK_ROOT/experiment/nohup_launcher"
mkdir -p "$F4_CAPTURE_DIR"
F4_CAPTURE="$F4_CAPTURE_DIR/F4_$(date +%Y%m%d_%H%M%S)_$$.log"
"$F4_PYTHON" -u repro/run_f4_screen_server.py "$@" 2>&1 | tee "$F4_CAPTURE"
# A failing Python/pipeline exits above; never schedule shutdown on failure.
F4_GROUP="$(sed -n 's/^F4 GROUP: //p' "$F4_CAPTURE" | tail -n 1)"
if [[ -z "$F4_GROUP" ]]; then
    echo 'Missing F4 group path: shutdown refused' >&2
    exit 1
fi
"$F4_PYTHON" -u repro/shutdown_f4_completed_run.py --group "$F4_GROUP" --invoke
