#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets}"
N12_RUN="${N12_RUN:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/experiment/all_runs/srpr_lowrank_audit}"

if [[ -z "$N12_RUN" ]]; then
  echo 'N12_RUN is required and must point to the N12 srprv2 run directory' >&2
  exit 2
fi
CHECKPOINT="$N12_RUN/model/model_20.pt"
REFERENCE="$N12_RUN/per_image_metrics/epoch_0020.pt"
for path in "$CHECKPOINT" "$REFERENCE"; do
  [[ -f "$path" ]] || { echo "Missing N12 artifact: $path" >&2; exit 2; }
done
[[ -f "$DATA_ROOT/DIV2K/DIV2K_valid_HR/0801.png" ]] || {
  echo "Missing DIV2K validation data under $DATA_ROOT" >&2
  exit 2
}
[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "Output already exists: $OUTPUT_DIR" >&2
  exit 2
}
mkdir -p "$OUTPUT_DIR"

SOURCE_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
cat > "$OUTPUT_DIR/manifest.json" <<EOF
{
  "audit": "stage-specific SRPR gate low-rank SVD",
  "source_commit": "$SOURCE_COMMIT",
  "n12_run": "$N12_RUN",
  "checkpoint": "$CHECKPOINT",
  "reference_metrics": "$REFERENCE",
  "data_root": "$DATA_ROOT",
  "screen_images": 20,
  "confirm_images": 100,
  "training": false
}
EOF

"$PYTHON_BIN" "$SCRIPT_DIR/check_srpr_lowrank_audit.py" \
  | tee "$OUTPUT_DIR/structure_check.json"

"$PYTHON_BIN" "$SCRIPT_DIR/audit_srpr_lowrank.py" \
  --checkpoint "$CHECKPOINT" \
  --reference-metrics "$REFERENCE" \
  --data-root "$DATA_ROOT" \
  --screen-images 20 \
  --confirm-images 100 \
  --workers "${WORKERS:-4}" \
  --output "$OUTPUT_DIR/srpr_lowrank_audit.json" \
  | tee "$OUTPUT_DIR/console.log"

cat "$OUTPUT_DIR/srpr_lowrank_audit.txt"
