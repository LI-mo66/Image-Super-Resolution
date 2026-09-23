#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash repro/run_n14_d01_server.sh CHECKPOINT [RUN_NAME]" >&2
  exit 2
fi

CHECKPOINT="$(realpath "$1")"
RUN_NAME="${2:-n14_d01_$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets}"
OUTPUT="$PROJECT_ROOT/experiment/all_runs/$RUN_NAME"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$DATA_ROOT/DIV2K/DIV2K_train_HR/0001.png" ]]; then
  echo "DIV2K not found under $DATA_ROOT" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "Output already exists: $OUTPUT" >&2
  exit 2
fi

mkdir -p "$OUTPUT"
git -C "$PROJECT_ROOT" rev-parse HEAD | tee "$OUTPUT/commit.txt"
printf 'checkpoint=%s\ndata_root=%s\npairing=homologous\n' \
  "$CHECKPOINT" "$DATA_ROOT" | tee "$OUTPUT/protocol.txt"
sha256sum "$CHECKPOINT" | tee "$OUTPUT/checkpoint.sha256"

python "$SCRIPT_DIR/n14_d0.py" smoke \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT/smoke.json" \
  2>&1 | tee "$OUTPUT/smoke.log"

python "$SCRIPT_DIR/n14_d0.py" operators \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT/homologous_operators.json" \
  --pairing homologous \
  --patch 32 \
  --operator-images 100 \
  --ranks 8,16,24,32 \
  2>&1 | tee "$OUTPUT/homologous_operators.log"

python "$SCRIPT_DIR/summarize_n14_d01.py" "$OUTPUT" | tee "$OUTPUT/summary.txt"
printf '\nN14 D0.1 outputs: %s\n' "$OUTPUT"
