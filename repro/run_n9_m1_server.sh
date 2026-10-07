#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
exec "${PYTHON:-python}" "$ROOT/repro/run_n9_m1_server.py" "$@"
