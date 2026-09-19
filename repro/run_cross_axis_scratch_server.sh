#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINE="${1:?Pass the absolute path to the completed 150-epoch scratch baseline}"
RUN_NAME="${2:-scratch/cross_axis_20e_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets}"
OUTPUT="$PROJECT_ROOT/experiment/all_runs/$RUN_NAME"

BASELINE="$(cd "$BASELINE" && pwd)"
if [[ -e "$OUTPUT" ]]; then
  printf 'Output already exists: %s\n' "$OUTPUT" >&2
  exit 2
fi
if [[ ! -f "$DATA_ROOT/DIV2K/DIV2K_train_HR/0001.png" ]]; then
  printf 'DIV2K training data missing under %s\n' "$DATA_ROOT" >&2
  exit 2
fi

python "$SCRIPT_DIR/summarize_cross_axis_scratch.py" "$BASELINE"
python "$SCRIPT_DIR/check_cross_axis.py"
mkdir -p "$OUTPUT"
(
  cd "$PROJECT_ROOT/LFMN"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python main.py \
    --dir_data "$DATA_ROOT" \
    --model LFMNCrossAxis \
    --data_train DIV2K \
    --data_test DIV2K \
    --data_range '1-800/801-900' \
    --scale 4 \
    --patch_size 256 \
    --batch_size 4 \
    --n_threads 8 \
    --ext img \
    --epochs 20 \
    --test_every 1000 \
    --lr 2e-4 \
    --scheduler cosine \
    --scheduler_t_max 150 \
    --eta_min 1e-6 \
    --loss '1*L1' \
    --seed 1 \
    --print_every 100 \
    --save "$RUN_NAME"
) 2>&1 | tee "$OUTPUT/console.log"

python "$SCRIPT_DIR/summarize_cross_axis_scratch.py" "$BASELINE" \
  --candidate "$OUTPUT" | tee "$OUTPUT/summary.txt"
printf '\nCandidate output: %s\nReused baseline curves: %s\n' "$OUTPUT" "$BASELINE"
