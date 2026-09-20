#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_NAME="${1:-scratch/n9_pcstr_20e_seed1_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets}"
OUTPUT="$PROJECT_ROOT/experiment/all_runs/$RUN_NAME"

if [[ -e "$OUTPUT" ]]; then
  printf 'Output already exists: %s\n' "$OUTPUT" >&2
  exit 2
fi
if [[ ! -f "$DATA_ROOT/DIV2K/DIV2K_train_HR/0001.png" ]]; then
  printf 'DIV2K training data missing under %s\n' "$DATA_ROOT" >&2
  exit 2
fi

python "$SCRIPT_DIR/check_pcstr.py"
mkdir -p "$OUTPUT"

run_one() {
  local name="$1"
  local model="$2"
  local run="$RUN_NAME/$name"
  local run_output="$OUTPUT/$name"
  mkdir -p "$run_output"
  printf '\n=== %s (%s) ===\n' "$name" "$model"
  (
    cd "$PROJECT_ROOT/LFMN"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python main.py \
      --dir_data "$DATA_ROOT" \
      --model "$model" \
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
      --save_per_image_metrics \
      --save "$run"
  ) 2>&1 | tee "$run_output/console.log"
}

run_one baseline LFMN
run_one pcstr LFMNPCSTR
run_one pcstr_wide LFMNPCSTRWide
run_one mix_control LFMNMixControl

python "$SCRIPT_DIR/summarize_n9_scratch.py" "$OUTPUT" \
  --data-root "$DATA_ROOT" | tee "$OUTPUT/summary.txt"
printf '\nN9 output: %s\n' "$OUTPUT"
