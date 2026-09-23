#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: bash repro/run_n12_causal_audit_server.sh SRPR_RUN B0_E20_RUN B0_E40_RUN [RUN_NAME]" >&2
  exit 2
fi

SRPR_RUN="$(realpath "$1")"
B0_E20_RUN="$(realpath "$2")"
B0_E40_RUN="$(realpath "$3")"
RUN_NAME="${4:-n12_causal_audit_$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets}"
OUTPUT="$PROJECT_ROOT/experiment/all_runs/$RUN_NAME"

declare -a CHECKPOINTS=(
  "$SRPR_RUN/model/model_20.pt"
  "$SRPR_RUN/model/model_40.pt"
  "$B0_E20_RUN/model/model_20.pt"
  "$B0_E40_RUN/model/model_40.pt"
)
for checkpoint in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint not found: $checkpoint" >&2
    exit 2
  fi
done
if [[ ! -f "$DATA_ROOT/DIV2K/DIV2K_valid_HR/0801.png" ]]; then
  echo "DIV2K validation data not found under $DATA_ROOT" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "Output already exists: $OUTPUT" >&2
  exit 2
fi

mkdir -p "$OUTPUT"
git -C "$PROJECT_ROOT" rev-parse HEAD | tee "$OUTPUT/commit.txt"
printf 'srpr_run=%s\nb0_e20_run=%s\nb0_e40_run=%s\ndata_root=%s\n' \
  "$SRPR_RUN" "$B0_E20_RUN" "$B0_E40_RUN" "$DATA_ROOT" | tee "$OUTPUT/protocol.txt"
sha256sum "${CHECKPOINTS[@]}" | tee "$OUTPUT/checkpoints.sha256"

python "$SCRIPT_DIR/audit_n12_srprv2_causality.py" \
  --srpr-checkpoint "$SRPR_RUN/model/model_20.pt" \
  --baseline-checkpoint "$B0_E20_RUN/model/model_20.pt" \
  --data-root "$DATA_ROOT" \
  --dataset DIV2K \
  --limit 100 \
  --output "$OUTPUT/epoch_0020.json" \
  2>&1 | tee "$OUTPUT/epoch_0020.log"

python "$SCRIPT_DIR/audit_n12_srprv2_causality.py" \
  --srpr-checkpoint "$SRPR_RUN/model/model_40.pt" \
  --baseline-checkpoint "$B0_E40_RUN/model/model_40.pt" \
  --data-root "$DATA_ROOT" \
  --dataset DIV2K \
  --limit 100 \
  --output "$OUTPUT/epoch_0040.json" \
  2>&1 | tee "$OUTPUT/epoch_0040.log"

python "$SCRIPT_DIR/summarize_n12_causal_audit.py" "$OUTPUT" | tee "$OUTPUT/summary.txt"
printf '\nN12 causal-audit outputs: %s\n' "$OUTPUT"
