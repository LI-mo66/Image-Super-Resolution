#!/usr/bin/env bash
# Run under nohup + setsid on a dedicated AutoDL instance.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p experiment
exec 9>experiment/n21_n22_gpu0.lock
if ! flock -n 9; then
  echo "Another N21/N22 queue owns this repository lock" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python repro/run_n21_n22_screen_server.py "$@"
