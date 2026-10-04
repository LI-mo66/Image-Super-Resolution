#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_NAME="${1:-n16/champion_srpr_c1_1000e_seed1}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets}"
SWINIR_REPO="${SWINIR_REPO:-$PROJECT_ROOT/repro/swinir_ref}"
TEACHER_CHECKPOINT="${RGCRD_TEACHER_CHECKPOINT:-$PROJECT_ROOT/repro/teacher_weights/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"

"$PYTHON_BIN" "$SCRIPT_DIR/check_n16_diagnostic_fastpath.py"
"$PYTHON_BIN" "$SCRIPT_DIR/check_n16_srpr_c1.py"
"$PYTHON_BIN" "$SCRIPT_DIR/check_rgcrd_teacher.py" \
  --repo "$SWINIR_REPO" --checkpoint "$TEACHER_CHECKPOINT" --teacher-only
"$PYTHON_BIN" "$SCRIPT_DIR/check_div2k_x4.py" \
  --root "$DATA_ROOT/DIV2K"

"$PYTHON_BIN" "$SCRIPT_DIR/run_n16_champion_1000e_server.py" \
  --run-name "$RUN_NAME" \
  --data-root "$DATA_ROOT" \
  --teacher-repo "$SWINIR_REPO" \
  --teacher-checkpoint "$TEACHER_CHECKPOINT" \
  --python-bin "$PYTHON_BIN" \
  --gpu "$GPU"

TRAIN_OUTPUT="$PROJECT_ROOT/experiment/all_runs/$RUN_NAME"
BENCHMARK_OUTPUT="${TRAIN_OUTPUT}_benchmarks"
"$PYTHON_BIN" "$SCRIPT_DIR/run_n16_champion_benchmarks_server.py" \
  --source "$TRAIN_OUTPUT" \
  --output "$BENCHMARK_OUTPUT" \
  --data-root "$DATA_ROOT" \
  --python-bin "$PYTHON_BIN" \
  --gpu "$GPU" \
  --workers 8

printf '\nN16 frozen champion output: %s/experiment/all_runs/%s\n' \
  "$PROJECT_ROOT" "$RUN_NAME"
printf 'N16 fixed benchmark output: %s\n' "$BENCHMARK_OUTPUT"
