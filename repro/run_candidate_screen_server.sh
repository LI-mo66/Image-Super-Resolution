#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-5}"
VALIDATION_END="${2:-810}"
LEARNING_RATE="${3:-1e-5}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="screen/candidates_${STAMP}"

run_one() {
  local name="$1"
  local model="$2"
  local loss_spec="$3"
  local freq_lr_mult="$4"
  echo "=== ${name}: model=${model}; loss=${loss_spec} ==="
  bash "${SCRIPT_DIR}/train_x4_server.sh" \
    finetune 0 "$EPOCHS" "$VALIDATION_END" "${ROOT}/${name}" \
    "$model" 3-5-7 8 "$LEARNING_RATE" 1 "$loss_spec" "$freq_lr_mult"
}

echo "Matched candidate screening: ${ROOT}"
echo "Epochs=${EPOCHS}; validation=0801-${VALIDATION_END}; lr=${LEARNING_RATE}"

# One matched baseline, then one variable at a time.
run_one baseline LFMN '1*L1' 1
run_one overlap_control LFMNOverlap '1*L1' 1
run_one shared_frequency LFMNFreq '1*L1' 10
run_one hf_loss LFMN '1*L1+0.05*HFL1' 1

echo "Candidate screening completed under experiment/all_runs/${ROOT}/"
python "${SCRIPT_DIR}/summarize_candidate_screen.py" \
  "${SCRIPT_DIR}/../experiment/all_runs/${ROOT}"
