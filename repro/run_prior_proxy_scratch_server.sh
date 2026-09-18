#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-150}"
VALIDATION_END="${2:-900}"
LEARNING_RATE="${3:-2e-4}"
SCHEDULER="${4:-cosine}"
ETA_MIN="${5:-1e-6}"
SEED="${6:-1}"
RUN_NAME="${7:-prior_proxy_scratch_$(date +%Y%m%d_%H%M%S)}"

if (( EPOCHS < 1 )); then
  echo 'EPOCHS must be positive.' >&2
  exit 2
fi
if (( VALIDATION_END < 801 || VALIDATION_END > 900 )); then
  echo 'VALIDATION_END must be between 801 and 900.' >&2
  exit 2
fi
if [[ "$SCHEDULER" != 'multistep' && "$SCHEDULER" != 'cosine' ]]; then
  echo 'SCHEDULER must be multistep or cosine.' >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets}"
OUTPUT="$PROJECT_ROOT/experiment/all_runs/$RUN_NAME"
if [[ -e "$OUTPUT" ]]; then
  echo "Output already exists: $OUTPUT" >&2
  exit 2
fi
if [[ ! -f "$DATA_ROOT/DIV2K/DIV2K_train_HR/0001.png" ]]; then
  echo "DIV2K training data not found under $DATA_ROOT" >&2
  exit 2
fi

run_one() {
  local model="$1"
  local name="$2"
  local path="$OUTPUT/$name"
  mkdir -p "$path"
  local args=(
    main.py
    --dir_data "$DATA_ROOT"
    --model "$model"
    --data_train DIV2K
    --data_test DIV2K
    --data_range "1-800/801-$VALIDATION_END"
    --scale 4
    --patch_size 256
    --batch_size 4
    --n_threads 8
    --ext img
    --epochs "$EPOCHS"
    --test_every 1000
    --lr "$LEARNING_RATE"
    --scheduler "$SCHEDULER"
    --eta_min "$ETA_MIN"
    --decay 200-400-600-800
    --gamma 0.5
    --loss '1*L1'
    --seed "$SEED"
    --print_every 100
    --save "$RUN_NAME/$name"
  )
  echo "Running $model -> $path"
  (
    cd "$PROJECT_ROOT/LFMN"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python "${args[@]}"
  ) 2>&1 | tee "$path/console.log"
}

mkdir -p "$OUTPUT"
printf 'Proxy scratch run: epochs=%s validation=0801-%s lr=%s scheduler=%s eta_min=%s seed=%s\n' \
  "$EPOCHS" "$VALIDATION_END" "$LEARNING_RATE" "$SCHEDULER" "$ETA_MIN" "$SEED" | tee "$OUTPUT/protocol.txt"
run_one LFMN baseline
run_one LFMNPriorProxy prior_proxy
python "$SCRIPT_DIR/summarize_prior_proxy.py" "$OUTPUT" | tee "$OUTPUT/summary.txt"
printf '\nOutput: %s\n' "$OUTPUT"
