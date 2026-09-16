#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCREEN_DIR="${1:-}"
CANDIDATE="${2:-rdsm_full}"
EPOCH="${3:-5}"
VALIDATION_END="${4:-810}"

if [[ -z "$SCREEN_DIR" ]]; then
  SCREEN_DIR="$(find "$PROJECT_ROOT/experiment/all_runs/screen" \
    -maxdepth 1 -type d -name 'rdsm_*' | sort | tail -n 1)"
fi
if [[ -z "$SCREEN_DIR" || ! -d "$SCREEN_DIR/$CANDIDATE" ]]; then
  echo "RDSM run not found: ${SCREEN_DIR:-<none>}/$CANDIDATE" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
python repro/inspect_rdsm_dynamics.py \
  --run-dir "$SCREEN_DIR/$CANDIDATE" \
  --data-root "${DATA_ROOT:-$PROJECT_ROOT/datasets}" \
  --start 801 \
  --end "$VALIDATION_END" \
  --epoch "$EPOCH"
