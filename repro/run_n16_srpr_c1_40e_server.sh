#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_20="${SOURCE_20:-}"
RUN_NAME="${1:-n16/srpr_c1_40e_seed1_$(date +%Y%m%d_%H%M%S)}"
OUTPUT="$PROJECT_ROOT/experiment/all_runs/$RUN_NAME"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets}"
SWINIR_REPO="${SWINIR_REPO:-$PROJECT_ROOT/repro/swinir_ref}"
TEACHER_CHECKPOINT="${RGCRD_TEACHER_CHECKPOINT:-$PROJECT_ROOT/repro/teacher_weights/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
TEACHER_MICROBATCH="${TEACHER_MICROBATCH:-1}"

if [[ -z "$SOURCE_20" ]]; then
  printf 'SOURCE_20 must point to the completed N16 20-epoch directory.\n' >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  printf 'Continuation output already exists: %s\n' "$OUTPUT" >&2
  exit 2
fi

"$PYTHON_BIN" "$SCRIPT_DIR/check_n16_40e_resume.py"
"$PYTHON_BIN" "$SCRIPT_DIR/check_n16_40e_summary.py"
"$PYTHON_BIN" "$SCRIPT_DIR/check_n16_srpr_c1.py"
"$PYTHON_BIN" "$SCRIPT_DIR/check_rgcrd_teacher.py" \
  --repo "$SWINIR_REPO" --checkpoint "$TEACHER_CHECKPOINT" --teacher-only

"$PYTHON_BIN" "$SCRIPT_DIR/run_n16_srpr_c1_40e_server.py" \
  --source "$SOURCE_20" \
  --output "$OUTPUT" \
  --data-root "$DATA_ROOT" \
  --teacher-repo "$SWINIR_REPO" \
  --teacher-checkpoint "$TEACHER_CHECKPOINT" \
  --teacher-microbatch "$TEACHER_MICROBATCH" \
  --python-bin "$PYTHON_BIN" \
  --gpu "$GPU"

printf '\nN16 matched 40-epoch continuation output: %s\n' "$OUTPUT"
