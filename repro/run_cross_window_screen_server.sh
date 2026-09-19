#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINE="${1:?Pass the absolute path to the existing matched baseline directory}"
RUN_NAME="${2:-screen/cross_window_$(date +%Y%m%d_%H%M%S)}"
OUTPUT="$PROJECT_ROOT/experiment/all_runs/$RUN_NAME"

BASELINE="$(cd "$BASELINE" && pwd)"
if [[ -e "$OUTPUT" ]]; then
  printf 'Output already exists: %s\n' "$OUTPUT" >&2
  exit 2
fi

python "$SCRIPT_DIR/summarize_cross_window.py" "$BASELINE"
python "$SCRIPT_DIR/check_cross_window.py"
mkdir -p "$OUTPUT"

bash "$SCRIPT_DIR/train_x4_server.sh" \
  finetune 0 5 810 "$RUN_NAME" \
  LFMNCrossWindow 3-5-7 8 1e-5 1 '1*L1' 1 1 state 1 10 \
  2>&1 | tee "$OUTPUT/console.log"

python "$SCRIPT_DIR/summarize_cross_window.py" "$BASELINE" \
  --candidate "$OUTPUT" | tee "$OUTPUT/summary.txt"
printf '\nCandidate output: %s\nReused baseline: %s\n' "$OUTPUT" "$BASELINE"
