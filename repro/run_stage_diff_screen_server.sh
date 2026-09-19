#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINE="${1:?Pass the absolute path to the existing matched baseline directory}"
RUN_NAME="${2:-screen/stage_diff_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${3:-5}"
VALIDATION_END="${4:-810}"
LEARNING_RATE="${5:-1e-5}"
STAGE_DIFF_LR_MULT="${6:-10}"
OUTPUT="$PROJECT_ROOT/experiment/all_runs/$RUN_NAME"

BASELINE="$(cd "$BASELINE" && pwd)"
if [[ -e "$OUTPUT" ]]; then
  printf 'Output already exists; select a fresh run name: %s\n' "$OUTPUT" >&2
  exit 2
fi
if [[ "$EPOCHS" != '5' || "$VALIDATION_END" != '810' || \
      "$LEARNING_RATE" != '1e-5' || "$STAGE_DIFF_LR_MULT" != '10' ]]; then
  echo 'This reusable-baseline screen is fixed to 5 epochs, validation 810, lr 1e-5, gate multiplier 10.' >&2
  exit 2
fi

python "$SCRIPT_DIR/summarize_stage_diff.py" "$BASELINE"
python "$SCRIPT_DIR/check_stage_diff.py"
mkdir -p "$OUTPUT"
python "$SCRIPT_DIR/summarize_stage_diff.py" "$BASELINE" \
  --manifest "$OUTPUT/protocol.json"

bash "$SCRIPT_DIR/train_x4_server.sh" \
  finetune 0 "$EPOCHS" "$VALIDATION_END" "$RUN_NAME" \
  LFMNStageDiff 3-5-7 8 "$LEARNING_RATE" 1 '1*L1' 1 1 state \
  "$STAGE_DIFF_LR_MULT" 2>&1 | tee "$OUTPUT/console.log"

python "$SCRIPT_DIR/summarize_stage_diff.py" "$BASELINE" \
  --candidate "$OUTPUT" | tee "$OUTPUT/summary.txt"
printf '\nCandidate output: %s\nReused baseline: %s\n' "$OUTPUT" "$BASELINE"
