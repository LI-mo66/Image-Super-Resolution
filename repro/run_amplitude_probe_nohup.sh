#!/usr/bin/env bash
# Explicitly requested successful-completion shutdown; no training.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
F3_PYTHON="${F3_PYTHON:-python}"
"$F3_PYTHON" -u repro/probe_modulation_amplitude.py "$@"
GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
if printf '%s\n' "$GPU_PIDS" | grep -qE '^[[:space:]]*[0-9]+[[:space:]]*$'; then
    echo 'Other GPU compute job found: shutdown refused' >&2
    exit 1
fi
sync
echo 'Amplitude report saved. Requesting server shutdown.'
bash -c 'shutdown -h now'
