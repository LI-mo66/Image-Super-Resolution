#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SWINIR_REPO="${SWINIR_REPO:-$PROJECT_ROOT/repro/swinir_ref}"
SWINIR_COMMIT="${SWINIR_COMMIT:-6545850fbf8df298df73d81f3e8cba638787c8bd}"
WEIGHT_DIR="${RGCRD_WEIGHT_DIR:-$PROJECT_ROOT/repro/teacher_weights}"
CHECKPOINT="$WEIGHT_DIR/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"
EXPECTED_SHA256='129dc773ba2d4c07f3eb0bb116fbe692011b7cc072d9ca12797cd3748198610a'

python -m pip install -r "$PROJECT_ROOT/requirements-server.txt"
if [[ ! -d "$SWINIR_REPO/.git" ]]; then
  git clone --depth 1 https://github.com/JingyunLiang/SwinIR.git "$SWINIR_REPO"
fi
if ! git -C "$SWINIR_REPO" cat-file -e "$SWINIR_COMMIT^{commit}" 2>/dev/null; then
  git -C "$SWINIR_REPO" fetch --depth 1 origin "$SWINIR_COMMIT"
fi
git -C "$SWINIR_REPO" checkout --detach "$SWINIR_COMMIT"
mkdir -p "$WEIGHT_DIR"
if [[ ! -f "$CHECKPOINT" ]]; then
  python - "$CHECKPOINT" <<'PY'
import sys
import urllib.request
url = ('https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/'
       '001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth')
urllib.request.urlretrieve(url, sys.argv[1])
PY
fi
printf '%s  %s\n' "$EXPECTED_SHA256" "$CHECKPOINT" | sha256sum --check -
printf 'SwinIR source commit: '
git -C "$SWINIR_REPO" rev-parse HEAD
python "$SCRIPT_DIR/check_rgcrd.py"
python "$SCRIPT_DIR/check_rgcrd_teacher.py" \
  --repo "$SWINIR_REPO" --checkpoint "$CHECKPOINT"
printf 'RGCRD teacher ready: %s\n' "$CHECKPOINT"
