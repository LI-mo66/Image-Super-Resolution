#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-5}"
VALIDATION_END="${2:-810}"
LEARNING_RATE="${3:-1e-5}"
RDSM_LR_MULT="${4:-10}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="screen/rdsm_direct_${STAMP}"

run_one() {
  local name="$1"
  local model="$2"
  local loss_spec="$3"
  local rdsm_lr_mult="$4"
  echo "=== ${name}: model=${model}; loss=${loss_spec} ==="
  bash "${SCRIPT_DIR}/train_x4_server.sh" \
    finetune 0 "$EPOCHS" "$VALIDATION_END" "${ROOT}/${name}" \
    "$model" 3-5-7 8 "$LEARNING_RATE" 1 "$loss_spec" 1 \
    "$rdsm_lr_mult"
}

echo "RDSM-v2 matched screening: ${ROOT}"
echo "Epochs=${EPOCHS}; validation=0801-${VALIDATION_END}; lr=${LEARNING_RATE}"
python "${SCRIPT_DIR}/check_rdsm_direct.py"

# The baseline is rerun because the dedicated DataLoader generator is a
# correctness fix that changes the crop stream relative to the first screen.
run_one baseline LFMN '1*L1' 1
run_one direct_router LFMNRDSMDirect '1*L1' "$RDSM_LR_MULT"
run_one direct_full LFMNRDSMDirect '1*L1+0.05*RRES+0.05*RDEM' "$RDSM_LR_MULT"

echo "RDSM-v2 screening completed under experiment/all_runs/${ROOT}/"
python "${SCRIPT_DIR}/summarize_candidate_screen.py" \
  "${SCRIPT_DIR}/../experiment/all_runs/${ROOT}" \
  --names baseline direct_router direct_full
