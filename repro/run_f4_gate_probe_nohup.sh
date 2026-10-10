#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
mkdir -p experiment/nohup_launcher
CAPTURE="experiment/nohup_launcher/F4_gate_$(date +%Y%m%d_%H%M%S)_$$.log"
python -u repro/probe_f4_gates.py "$@" 2>&1 | tee "$CAPTURE"
OUT="$(sed -n 's/^GATE OUTPUT: //p' "$CAPTURE" | tail -n 1)"
test -n "$OUT"
printf '%s\n' "$OUT" > experiment/nohup_launcher/F4_gate_latest_output.txt
python -u repro/shutdown_f4_gate_probe.py --output "$OUT" --invoke
