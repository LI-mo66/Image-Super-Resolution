#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_NAME="${1:-n11_srprv1_20e_seed1_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets}"
OUTPUT="$PROJECT_ROOT/experiment/all_runs/$RUN_NAME"
PYTHON_BIN="${PYTHON_BIN:-python}"
B0_REFERENCE="${B0_REFERENCE:-}"

if [[ -z "$B0_REFERENCE" ]]; then
  printf 'B0_REFERENCE must point to the existing N9 from-scratch LFMN B0 directory\n' >&2
  exit 2
fi
if [[ "$B0_REFERENCE" != /* ]]; then
  B0_REFERENCE="$PROJECT_ROOT/experiment/all_runs/$B0_REFERENCE"
fi
if [[ ! -f "$B0_REFERENCE/config.txt" ]]; then
  printf 'N9 from-scratch B0 config missing: %s\n' "$B0_REFERENCE/config.txt" >&2
  exit 2
fi
for required in \
  "$B0_REFERENCE/psnr_log.pt" \
  "$B0_REFERENCE/ssim_log.pt" \
  "$B0_REFERENCE/model/model_20.pt" \
  "$B0_REFERENCE/per_image_metrics/epoch_0020.pt"; do
  if [[ ! -f "$required" ]]; then
    printf 'Required N9 B0 artifact missing: %s\n' "$required" >&2
    exit 2
  fi
done
for expected in \
  'model: LFMN' \
  'pre_train: ' \
  'data_range: 1-800/801-900' \
  'scale: [4]' \
  'patch_size: 256' \
  'batch_size: 4' \
  'seed: 1' \
  'epochs: 20' \
  'lr: 0.0002' \
  'scheduler: cosine' \
  'scheduler_t_max: 150' \
  'eta_min: 1e-06' \
  'loss: 1*L1'; do
  if ! grep -Fxq "$expected" "$B0_REFERENCE/config.txt"; then
    printf 'N9 B0 protocol mismatch or missing field: %s\n' "$expected" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT" ]]; then
  printf 'Output already exists: %s\n' "$OUTPUT" >&2
  exit 2
fi
if [[ ! -f "$DATA_ROOT/DIV2K/DIV2K_train_HR/0001.png" ]]; then
  printf 'DIV2K training data missing under %s\n' "$DATA_ROOT" >&2
  exit 2
fi
"$PYTHON_BIN" "$SCRIPT_DIR/check_n11.py"
mkdir -p "$OUTPUT"
"$PYTHON_BIN" "$SCRIPT_DIR/profile_n11.py" \
  --size 64 --warmup "${PROFILE_WARMUP:-10}" --repeats "${PROFILE_REPEATS:-50}" \
  --output "$OUTPUT/profile_metrics.json" | tee "$OUTPUT/profile_console.log"

cat > "$OUTPUT/run_manifest.json" <<EOF
{
  "candidate": "N11/SRPRv1",
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
  "eta_min": 0.000001,
  "mac_definition": "1 MAC = one multiply-accumulate; FLOPs = 2 * MAC",
  "input_shape_for_profile": "1x3x64x64"
}
EOF
(
  cd "$PROJECT_ROOT/LFMN"
  CUDA_VISIBLE_DEVICES="${GPU:-0}" PYTHONUNBUFFERED=1 "$PYTHON_BIN" main.py \
    --dir_data "$DATA_ROOT" \
    --model LFMNSRPRV1 \
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
    --save "$RUN_NAME/srprv1"
) 2>&1 | tee "$OUTPUT/console.log"
"$PYTHON_BIN" "$SCRIPT_DIR/summarize_n11.py" "$OUTPUT" \
  --baseline-reference "$B0_REFERENCE" \
  | tee "$OUTPUT/summary.txt"
printf '\nN11/SRPRv1 output: %s\n' "$OUTPUT"
