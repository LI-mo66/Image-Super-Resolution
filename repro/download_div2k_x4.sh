#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_DIR="$PROJECT_ROOT/datasets/_downloads"
DIV2K_DIR="$PROJECT_ROOT/datasets/DIV2K"
BASE_URL="https://data.vision.ee.ethz.ch/cvl/DIV2K"

for command_name in curl unzip python; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing command: $command_name" >&2
    echo "Install prerequisites with: apt-get update && apt-get install -y curl unzip" >&2
    exit 2
  fi
done

mkdir -p "$DOWNLOAD_DIR" "$DIV2K_DIR"

archives=(
  DIV2K_train_HR.zip
  DIV2K_train_LR_bicubic_X4.zip
  DIV2K_valid_HR.zip
  DIV2K_valid_LR_bicubic_X4.zip
)

download_archive() {
  local archive="$1"
  local destination="$DOWNLOAD_DIR/$archive"
  if [[ -f "$destination" ]] && unzip -tq "$destination" >/dev/null 2>&1; then
    echo "Archive already valid: $archive"
    return
  fi
  echo "Downloading: $archive"
  curl -L --fail --retry 5 --retry-delay 5 -C - \
    -o "$destination" "$BASE_URL/$archive"
  unzip -tq "$destination" >/dev/null
}

for archive in "${archives[@]}"; do
  download_archive "$archive"
done

declare -A markers=(
  [DIV2K_train_HR.zip]="DIV2K_train_HR/0001.png"
  [DIV2K_train_LR_bicubic_X4.zip]="DIV2K_train_LR_bicubic/X4/0001x4.png"
  [DIV2K_valid_HR.zip]="DIV2K_valid_HR/0801.png"
  [DIV2K_valid_LR_bicubic_X4.zip]="DIV2K_valid_LR_bicubic/X4/0801x4.png"
)

for archive in "${archives[@]}"; do
  if [[ -f "$DIV2K_DIR/${markers[$archive]}" ]]; then
    echo "Already extracted: $archive"
  else
    echo "Extracting: $archive"
    unzip -q -o "$DOWNLOAD_DIR/$archive" -d "$DIV2K_DIR"
  fi
done

cd "$PROJECT_ROOT"
python repro/check_div2k_x4.py