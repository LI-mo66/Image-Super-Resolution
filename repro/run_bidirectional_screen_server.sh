#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-5}"
VALIDATION_END="${2:-810}"
LEARNING_RATE="${3:-1e-5}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="screen/bidirectional_${STAMP}"

run_one() {
  local name="$1" model="$2"
  echo "=== ${name}: model=${model} ==="
  bash "${SCRIPT_DIR}/train_x4_server.sh" \
    finetune 0 "$EPOCHS" "$VALIDATION_END" "${ROOT}/${name}" \
    "$model" 3-5-7 8 "$LEARNING_RATE" 1 '1*L1' 1
}

echo "Bidirectional matched screening: ${ROOT}"
echo "Epochs=${EPOCHS}; validation=0801-${VALIDATION_END}; lr=${LEARNING_RATE}"
python "${SCRIPT_DIR}/check_bidirectional.py"
run_one baseline LFMN
run_one bidirectional LFMNBidirectional

echo "Bidirectional screening completed under experiment/all_runs/${ROOT}/"
python "${SCRIPT_DIR}/summarize_candidate_screen.py" \
  "${SCRIPT_DIR}/../experiment/all_runs/${ROOT}" \
  --names baseline bidirectional
