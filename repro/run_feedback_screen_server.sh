#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-10}"
VALIDATION_END="${2:-810}"
LEARNING_RATE="${3:-1e-5}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASELINE_RUN="screen/feedback_${STAMP}/baseline_ft_${EPOCHS}e"
FEEDBACK_RUN="screen/feedback_${STAMP}/feedback_s357_m8_ft_${EPOCHS}e"

echo "Matched screening experiment"
echo "Baseline: ${BASELINE_RUN}"
echo "Feedback: ${FEEDBACK_RUN}"
echo "Epochs: ${EPOCHS}; validation: 0801-${VALIDATION_END}; lr: ${LEARNING_RATE}"

bash "${SCRIPT_DIR}/train_x4_server.sh" finetune 0 "${EPOCHS}" "${VALIDATION_END}" "${BASELINE_RUN}" LFMN 3-5-7 8 "${LEARNING_RATE}"
bash "${SCRIPT_DIR}/train_x4_server.sh" finetune 0 "${EPOCHS}" "${VALIDATION_END}" "${FEEDBACK_RUN}" LFMNFeedback 3-5-7 8 "${LEARNING_RATE}"

echo "Matched screening completed."
echo "Compare logs under experiment/all_runs/screen/feedback_${STAMP}/"
