#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-5}"
VALIDATION_END="${2:-810}"
LEARNING_RATE="${3:-1e-5}"
RDSM_LR_MULT="${4:-10}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="screen/rdsm_${STAMP}"

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

echo "RDSM matched screening: ${ROOT}"
echo "Epochs=${EPOCHS}; validation=0801-${VALIDATION_END}; lr=${LEARNING_RATE}"

# These four runs isolate ordinary fine-tuning, unsupervised routing,
# intermediate residual supervision, and the full demand-distillation design.
run_one baseline LFMN '1*L1' 1
run_one router_only LFMNRDSM '1*L1' "$RDSM_LR_MULT"
run_one residual_supervision LFMNRDSM '1*L1+0.05*RRES' "$RDSM_LR_MULT"
run_one rdsm_full LFMNRDSM '1*L1+0.05*RRES+0.05*RDEM' "$RDSM_LR_MULT"

echo "RDSM screening completed under experiment/all_runs/${ROOT}/"
python "${SCRIPT_DIR}/summarize_candidate_screen.py" \
  "${SCRIPT_DIR}/../experiment/all_runs/${ROOT}" \
  --names baseline router_only residual_supervision rdsm_full
