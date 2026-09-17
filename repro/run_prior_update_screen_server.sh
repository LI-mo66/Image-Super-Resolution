#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINE="${1:-$PROJECT_ROOT/experiment/all_runs/screen/bidirectional_20260917_120840/baseline}"
BASELINE="$(cd "$BASELINE" && pwd)"
EPOCHS="${3:-5}"
VALIDATION_END="${4:-810}"
LEARNING_RATE="${5:-1e-5}"
PRIOR_MODE="${6:-state}"
RUN_NAME="${2:-screen/prior_update_${PRIOR_MODE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT="$PROJECT_ROOT/experiment/all_runs/$RUN_NAME"
if [[ -e "$OUTPUT" ]]; then
  printf 'Output already exists; select a fresh run name: %s\n' "$OUTPUT" >&2
  exit 2
fi
python "$SCRIPT_DIR/summarize_prior_update.py" "$BASELINE"
python "$SCRIPT_DIR/check_prior_update.py"
mkdir -p "$OUTPUT"
python "$SCRIPT_DIR/summarize_prior_update.py" "$BASELINE" --manifest "$OUTPUT/protocol.json"
bash "$SCRIPT_DIR/train_x4_server.sh" finetune 0 5 810 "$RUN_NAME" \
  LFMNPriorUpdate 3-5-7 8 "$LEARNING_RATE" 1 '1*L1' 1 1 "$PRIOR_MODE" 2>&1 | tee "$OUTPUT/console.log"
python "$SCRIPT_DIR/summarize_prior_update.py" "$BASELINE" \
  --candidate "$OUTPUT" | tee "$OUTPUT/summary.txt"
printf '\nCandidate output: %s\nReused baseline: %s\n' "$OUTPUT" "$BASELINE"
