#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-LFMN}"
DIR_DATA="${DIR_DATA:-/home/huanzhujie/datasets}"
DATA_TEST="${DATA_TEST:-Set5+Set14+B100+Urban100}"
SAVE_ROOT="${SAVE_ROOT:-test/lfmn_benchmark}"
SCALES="${SCALES:-2 3 4}"
SELF_ENSEMBLE="${SELF_ENSEMBLE:-1}"

SCALES="${SCALES//+/ }"
SCALES="${SCALES//,/ }"

ckpt_for_scale() {
    case "$1" in
        2) printf '%s\n' "$ROOT_DIR/model/scale2_model_996.pt" ;;
        3) printf '%s\n' "$ROOT_DIR/model/scale3_model_969.pt" ;;
        4) printf '%s\n' "$ROOT_DIR/model/scale4_model_939.pt" ;;
        *) printf 'Unsupported scale: %s\n' "$1" >&2; return 1 ;;
    esac
}

self_ensemble_args=()
case "$SELF_ENSEMBLE" in
    0|false|False|FALSE|no|No|NO) ;;
    *) self_ensemble_args=(--self_ensemble) ;;
esac

cd "$ROOT_DIR"

for scale in $SCALES; do
    checkpoint="$(ckpt_for_scale "$scale")"
    if [[ ! -f "$checkpoint" ]]; then
        printf 'Missing checkpoint for scale x%s: %s\n' "$scale" "$checkpoint" >&2
        exit 1
    fi

    printf '\n==> Test %s x%s\n' "$MODEL" "$scale"
    "$PYTHON_BIN" main.py \
        --dir_data "$DIR_DATA" \
        --model "$MODEL" \
        --data_test "$DATA_TEST" \
        --scale "$scale" \
        --pre_train "$checkpoint" \
        --test_only \
        "${self_ensemble_args[@]}" \
        --save "$SAVE_ROOT/x${scale}" \
        "$@"
done
