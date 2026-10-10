#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
F4_PYTHON="${F4_PYTHON:-python}"
mkdir -p experiment/nohup_launcher
CAPTURE="experiment/nohup_launcher/F4_150_$(date +%Y%m%d_%H%M%S)_$$.log"
"$F4_PYTHON" -u repro/run_f4_150_server.py "$@" 2>&1 | tee "$CAPTURE"
GROUP="$(sed -n 's/^F4 150 GROUP: //p' "$CAPTURE" | tail -n 1)"
if [[ -z "$GROUP" ]]; then
  echo 'No continuation group; shutdown refused' >&2
  exit 1
fi
printf '%s\n' "$GROUP" > experiment/nohup_launcher/F4_150_latest_group.txt
"$F4_PYTHON" -u repro/shutdown_f4_150.py --group "$GROUP" --invoke
