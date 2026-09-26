#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_NAME="${1:-n16/srpr_c1_20e_seed1_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets}"
SWINIR_REPO="${SWINIR_REPO:-$PROJECT_ROOT/repro/swinir_ref}"
TEACHER_CHECKPOINT="${RGCRD_TEACHER_CHECKPOINT:-$PROJECT_ROOT/repro/teacher_weights/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
GROUPS="${GROUPS:-c1 srpr_c1}"
TEACHER_MICROBATCH="${TEACHER_MICROBATCH:-1}"
OUTPUT="$PROJECT_ROOT/experiment/all_runs/$RUN_NAME"
SOURCE_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"

if [[ -e "$OUTPUT" ]]; then
  printf 'Output already exists: %s\n' "$OUTPUT" >&2
  exit 2
fi
if [[ ! -f "$DATA_ROOT/DIV2K/DIV2K_train_HR/0001.png" ]]; then
  printf 'DIV2K training data missing under %s\n' "$DATA_ROOT" >&2
  exit 2
fi
if [[ ! -f "$TEACHER_CHECKPOINT" || ! -f "$SWINIR_REPO/models/network_swinir.py" ]]; then
  printf 'Teacher missing. Run repro/setup_rgcrd_teacher_server.sh first.\n' >&2
  exit 2
fi
if [[ "$GROUPS" != "c1 srpr_c1" ]]; then
  printf 'N16 requires the complete paired groups: c1 srpr_c1\n' >&2
  exit 2
fi

mkdir -p "$OUTPUT"
"$PYTHON_BIN" "$SCRIPT_DIR/check_div2k_x4.py" \
  --root "$DATA_ROOT/DIV2K" | tee "$OUTPUT/div2k_check.log"
"$PYTHON_BIN" "$SCRIPT_DIR/check_n16_srpr_c1.py" \
  | tee "$OUTPUT/n16_structure_check.json"
"$PYTHON_BIN" "$SCRIPT_DIR/check_rgcrd_teacher.py" \
  --repo "$SWINIR_REPO" --checkpoint "$TEACHER_CHECKPOINT" --teacher-only \
  | tee "$OUTPUT/teacher_check.log"

cat > "$OUTPUT/run_manifest.json" <<EOF
{
  "candidate": "N16/SRPRv2+C1 interaction",
  "source_commit": "$SOURCE_COMMIT",
  "groups": ["c1", "srpr_c1"],
  "comparison": "SRPRv2+C1 minus B0+C1",
  "from_scratch": true,
  "epochs": 20,
  "data_range": "1-800/801-900",
  "scale": 4,
  "patch_size_hr": 256,
  "batch_size": 4,
  "seed": 1,
  "loss": "1*L1 + 0.1*output_KD",
  "optimizer": "ADAM",
  "learning_rate": 0.0002,
  "scheduler": "cosine",
  "scheduler_t_max": 150,
  "eta_min": 0.000001,
  "teacher_microbatch": $TEACHER_MICROBATCH
}
EOF

run_one() {
  local name="$1"
  local model="$2"
  local run="$RUN_NAME/$name"
  local run_output="$OUTPUT/$name"
  mkdir -p "$run_output"
  printf '\n=== %s (model=%s, output KD=0.1) ===\n' "$name" "$model"
  (
    cd "$PROJECT_ROOT/LFMN"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PYTHON_BIN" main.py \
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
      --rgcrd_mode output \
      --rgcrd_teacher_repo "$SWINIR_REPO" \
      --rgcrd_teacher_checkpoint "$TEACHER_CHECKPOINT" \
      --rgcrd_teacher_microbatch "$TEACHER_MICROBATCH" \
      --rgcrd_lambda_output 0.1 \
      --rgcrd_grad_diag_every 1 \
      --save "$run"
  ) 2>&1 | tee "$run_output/console.log"
}

run_one c1 LFMN
run_one srpr_c1 LFMNSRPRV2

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_n16_srpr_c1.py" "$OUTPUT" \
  | tee "$OUTPUT/summary.txt"
printf '\nN16 output: %s\n' "$OUTPUT"
