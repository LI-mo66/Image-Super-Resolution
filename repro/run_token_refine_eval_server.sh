#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINE="${1:?Pass the absolute path to the completed baseline directory}"
VALIDATION_END="${2:-810}"
RUN_NAME="${3:-screen/token_refine_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets}"
OUTPUT="$PROJECT_ROOT/experiment/all_runs/$RUN_NAME"

BASELINE="$(cd "$BASELINE" && pwd)"
CHECKPOINT="$BASELINE/model/model_5.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  printf 'Baseline epoch-5 checkpoint not found: %s\n' "$CHECKPOINT" >&2
  exit 2
fi
if (( VALIDATION_END < 801 || VALIDATION_END > 900 )); then
  echo 'VALIDATION_END must be between 801 and 900.' >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  printf 'Output already exists: %s\n' "$OUTPUT" >&2
  exit 2
fi

python "$SCRIPT_DIR/check_token_refine.py"
mkdir -p "$OUTPUT"

run_one() {
  local iterations="$1"
  local name="refine${iterations}"
  local save_name="$RUN_NAME/$name"
  mkdir -p "$OUTPUT/$name"
  (
    cd "$PROJECT_ROOT/LFMN"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python main.py \
      --dir_data "$DATA_ROOT" \
      --model LFMNTokenRefine \
      --token_refine_iters "$iterations" \
      --data_train DIV2K \
      --data_test DIV2K \
      --data_range "1-800/801-$VALIDATION_END" \
      --scale 4 \
      --n_threads 8 \
      --ext img \
      --pre_train "$CHECKPOINT" \
      --test_only \
      --save "$save_name"
  ) 2>&1 | tee "$OUTPUT/$name/console.log"
}

run_one 1
run_one 3
python "$SCRIPT_DIR/summarize_token_refine.py" \
  "$BASELINE" "$OUTPUT/refine1" "$OUTPUT/refine3" | tee "$OUTPUT/summary.txt"
printf '\nOutput: %s\nCheckpoint: %s\n' "$OUTPUT" "$CHECKPOINT"
