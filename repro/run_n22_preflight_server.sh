#!/usr/bin/env bash
# Engineering preflight ONLY: no screening/long training is started here.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$#" -ne 3 ]]; then
  echo "Usage: bash repro/run_n22_preflight_server.sh CHECKPOINT DIV2K_DIRECTORY NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
test -f "$1"
test -f "$2/DIV2K_train_HR/0001.png"
test -f "$2/DIV2K_valid_LR_bicubic/X4/0859x4.png"
git rev-parse HEAD
python repro/check_n22.py --checkpoint "$1" --data "$2" --output "$3"
python repro/summarize_n22_checks.py "$3/checks.json"
