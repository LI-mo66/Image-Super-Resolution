#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_NAME="${1:-rgcrd/screen_40e_seed1_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets}"
SWINIR_REPO="${SWINIR_REPO:-$PROJECT_ROOT/repro/swinir_ref}"
TEACHER_CHECKPOINT="${RGCRD_TEACHER_CHECKPOINT:-$PROJECT_ROOT/repro/teacher_weights/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth}"
EPOCHS="${EPOCHS:-40}"
GROUPS="${GROUPS:-b0 c1 m0 m1}"
GPU="${GPU:-0}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PATCH_SIZE="${PATCH_SIZE:-256}"
N_THREADS="${N_THREADS:-8}"
TEACHER_MICROBATCH="${TEACHER_MICROBATCH:-1}"
LAMBDA_OUTPUT="${LAMBDA_OUTPUT:-0.1}"
LAMBDA_REL="${LAMBDA_REL:-0.25}"
LAMBDA_EVO="${LAMBDA_EVO:-0.125}"
OUTPUT="$PROJECT_ROOT/experiment/all_runs/$RUN_NAME"

if [[ -e "$OUTPUT" && "$ALLOW_EXISTING" != "1" ]]; then
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

python "$SCRIPT_DIR/check_rgcrd.py"
python "$SCRIPT_DIR/check_rgcrd_teacher.py" \
  --repo "$SWINIR_REPO" --checkpoint "$TEACHER_CHECKPOINT"
mkdir -p "$OUTPUT"

run_one() {
  local name="$1"
  local model="$2"
  local mode="$3"
  local run="$RUN_NAME/$name"
  local run_output="$OUTPUT/$name"
  if [[ -e "$run_output" ]]; then
    printf 'Group output already exists: %s\n' "$run_output" >&2
    exit 2
  fi
  mkdir -p "$run_output"
  printf '\n=== %s (model=%s, RGCRD=%s) ===\n' "$name" "$model" "$mode"
  (
    cd "$PROJECT_ROOT/LFMN"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 python main.py \
      --dir_data "$DATA_ROOT" \
      --model "$model" \
      --data_train DIV2K \
      --data_test DIV2K \
      --data_range '1-800/801-900' \
      --scale 4 \
      --patch_size "$PATCH_SIZE" \
      --batch_size "$BATCH_SIZE" \
      --n_threads "$N_THREADS" \
      --ext img \
      --epochs "$EPOCHS" \
      --test_every 1000 \
      --lr 2e-4 \
      --scheduler multistep \
      --decay '200-400-600-800' \
      --gamma 0.5 \
      --loss '1*L1' \
      --seed 1 \
      --print_every 100 \
      --save_per_image_metrics \
      --rgcrd_mode "$mode" \
      --rgcrd_teacher_repo "$SWINIR_REPO" \
      --rgcrd_teacher_checkpoint "$TEACHER_CHECKPOINT" \
      --rgcrd_teacher_microbatch "$TEACHER_MICROBATCH" \
      --rgcrd_lambda_output "$LAMBDA_OUTPUT" \
      --rgcrd_lambda_rel "$LAMBDA_REL" \
      --rgcrd_lambda_evo "$LAMBDA_EVO" \
      --rgcrd_local_windows '4+8' \
      --rgcrd_global_grid 8 \
      --rgcrd_reliability_pixel 0.5 \
      --rgcrd_reliability_low 0.25 \
      --rgcrd_reliability_grad 0.25 \
      --rgcrd_reliability_temperature 0.25 \
      --rgcrd_grad_diag_every 1 \
      --save "$run"
  ) 2>&1 | tee "$run_output/console.log"
}

for group in $GROUPS; do
  case "$group" in
    b0) run_one b0 LFMN off ;;
    c1) run_one c1 LFMN output ;;
    m0) run_one m0 LFMNRGCRD relation ;;
    m1) run_one m1 LFMNRGCRD full ;;
    *)
      printf 'Unknown RGCRD group: %s\n' "$group" >&2
      exit 2
      ;;
  esac
done

if [[ -f "$OUTPUT/b0/psnr_log.pt" \
   && -f "$OUTPUT/c1/psnr_log.pt" \
   && -f "$OUTPUT/m0/psnr_log.pt" \
   && -f "$OUTPUT/m1/psnr_log.pt" ]]; then
  python "$SCRIPT_DIR/summarize_rgcrd_screen.py" "$OUTPUT" | tee "$OUTPUT/summary.txt"
fi
printf '\nRGCRD screening output: %s\n' "$OUTPUT"
printf 'Compare the same epoch in psnr_log.pt; do not promote M1 from training loss alone.\n'
