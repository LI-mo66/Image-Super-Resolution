#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_NAME="${1:-n15/srprss_20e_seed1_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets}"
OUTPUT="$PROJECT_ROOT/experiment/all_runs/$RUN_NAME"
SOURCE_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
PYTHON_BIN="${PYTHON_BIN:-python}"
B0_REFERENCE="${B0_REFERENCE:-}"

if [[ -z "$B0_REFERENCE" ]]; then
  echo 'B0_REFERENCE is required' >&2
  exit 2
fi
for file in config.txt psnr_log.pt ssim_log.pt model/model_20.pt per_image_metrics/epoch_0020.pt; do
  [[ -f "$B0_REFERENCE/$file" ]] || { echo "Missing B0 artifact: $B0_REFERENCE/$file" >&2; exit 2; }
done
for expected in \
  'model: LFMN' 'pre_train: ' 'data_range: 1-800/801-900' \
  'scale: [4]' 'patch_size: 256' 'batch_size: 4' 'seed: 1' \
  'epochs: 20' 'lr: 0.0002' 'scheduler: cosine' \
  'scheduler_t_max: 150' 'eta_min: 1e-06' 'loss: 1*L1'; do
  grep -Fxq "$expected" "$B0_REFERENCE/config.txt" || {
    echo "B0 protocol mismatch: $expected" >&2
    exit 2
  }
done
[[ ! -e "$OUTPUT" ]] || { echo "Output exists: $OUTPUT" >&2; exit 2; }
[[ -f "$DATA_ROOT/DIV2K/DIV2K_train_HR/0001.png" ]] || {
  echo "Missing DATA_ROOT: $DATA_ROOT" >&2
  exit 2
}

mkdir -p "$OUTPUT"
"$PYTHON_BIN" "$SCRIPT_DIR/check_div2k_x4.py" --root "$DATA_ROOT/DIV2K"
"$PYTHON_BIN" "$SCRIPT_DIR/check_n15_srprss.py" \
  --root "$DATA_ROOT/DIV2K" | tee "$OUTPUT/n15_check.json"
"$PYTHON_BIN" "$SCRIPT_DIR/profile_n15_srprss.py" \
  --size 64 --warmup "${PROFILE_WARMUP:-10}" \
  --repeats "${PROFILE_REPEATS:-50}" \
  --output "$OUTPUT/profile_metrics.json" | tee "$OUTPUT/profile_console.log"

cat > "$OUTPUT/run_manifest.json" <<EOF
{
  "candidate": "N15/SRPR-SS",
  "source_commit": "$SOURCE_COMMIT",
  "baseline_reference": "$B0_REFERENCE",
  "comparison_mode": "from_scratch_paired_with_existing_n9_b0",
  "epochs": 20,
  "data_range": "1-800/801-900",
  "scale": 4,
  "patch_size_hr": 256,
  "batch_size": 4,
  "seed": 1,
  "loss": "1*L1",
  "optimizer": "ADAM",
  "learning_rate": 0.0002,
  "scheduler": "cosine",
  "scheduler_t_max": 150,
  "eta_min": 0.000001
}
EOF

(
  cd "$PROJECT_ROOT/LFMN"
  CUDA_VISIBLE_DEVICES="${GPU:-0}" PYTHONUNBUFFERED=1 "$PYTHON_BIN" main.py \
    --dir_data "$DATA_ROOT" --model LFMNSRPRSS \
    --data_train DIV2K --data_test DIV2K \
    --data_range '1-800/801-900' --scale 4 --patch_size 256 \
    --batch_size 4 --n_threads 8 --ext img --epochs 20 \
    --test_every 1000 --lr 2e-4 --scheduler cosine \
    --scheduler_t_max 150 --eta_min 1e-6 --loss '1*L1' \
    --seed 1 --print_every 100 --save_per_image_metrics \
    --save "$RUN_NAME/srprss"
) 2>&1 | tee "$OUTPUT/console.log"

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_n15_srprss.py" "$OUTPUT" \
  --baseline-reference "$B0_REFERENCE" | tee "$OUTPUT/summary.txt"
