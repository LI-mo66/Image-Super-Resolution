#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-$PWD/datasets}"
case "${1:-preflight}" in
  check-data) exec "$PYTHON_BIN" -u repro/original_lfmn_x4.py --data "$DATA_ROOT" --check-data ;;
  preflight) exec "$PYTHON_BIN" -u repro/preflight_original_lfmn_x4.py --data "$DATA_ROOT" ;;
  train)
    : "${PREFLIGHT_RECEIPT:?Set PREFLIGHT_RECEIPT to the passed receipt path}"
    exec "$PYTHON_BIN" -u repro/original_lfmn_x4.py --data "$DATA_ROOT" --preflight-receipt "$PREFLIGHT_RECEIPT" ;;
  resume)
    : "${RUN_DIR:?Set RUN_DIR to original formal run directory}"
    : "${PREFLIGHT_RECEIPT:?Set PREFLIGHT_RECEIPT to the original passed receipt path}"
    exec "$PYTHON_BIN" -u repro/original_lfmn_x4.py --data "$DATA_ROOT" --resume "$RUN_DIR" --preflight-receipt "$PREFLIGHT_RECEIPT" ;;
  *) echo 'Usage: bash repro/run_original_lfmn_x4_server.sh check-data|preflight|train|resume' >&2; exit 2 ;;
esac
