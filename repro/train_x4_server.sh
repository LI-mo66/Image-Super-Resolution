#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-scratch}"
MAX_TRAIN_BATCHES="${2:-20}"
EPOCHS="${3:-1}"
VALIDATION_END="${4:-810}"
RUN_NAME="${5:-}"
MODEL="${6:-LFMN}"
FEEDBACK_STAGES="${7:-3-5-7}"
FEEDBACK_MID="${8:-8}"
LEARNING_RATE="${9:-2e-4}"
FEEDBACK_LR_MULT="${10:-1}"
LOSS_SPEC="${11:-1*L1}"
FREQ_LR_MULT="${12:-1}"
RDSM_LR_MULT="${13:-1}"

if [[ "$MODE" != "scratch" && "$MODE" != "finetune" && "$MODE" != "resume" ]]; then
  echo "Mode must be scratch, finetune, or resume." >&2
  exit 2
fi
if (( VALIDATION_END < 801 || VALIDATION_END > 900 )); then
  echo "ValidationEnd must be between 801 and 900." >&2
  exit 2
fi
if (( MAX_TRAIN_BATCHES < 0 || EPOCHS < 1 )); then
  echo "MaxTrainBatches must be nonnegative and Epochs must be at least 1." >&2
  exit 2
fi
if [[ "$MODEL" != "LFMN" && "$MODEL" != "LFMNFeedback" && "$MODEL" != "LFMNFreq" && "$MODEL" != "LFMNOverlap" && "$MODEL" != "LFMNRDSM" && "$MODEL" != "LFMNRDSMDirect" && "$MODEL" != "LFMNBidirectional" && "$MODEL" != "LFMNPriorUpdate" ]]; then
  echo "Unsupported model: $MODEL" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LFMN_ROOT="$PROJECT_ROOT/LFMN"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets}"
PRETRAINED="$LFMN_ROOT/model/scale4_model_939.pt"

if [[ ! -f "$DATA_ROOT/DIV2K/DIV2K_train_HR/0001.png" ]]; then
  echo "DIV2K training data not found under $DATA_ROOT" >&2
  exit 2
fi

if [[ "$MODE" == "resume" ]]; then
  if [[ -z "$RUN_NAME" ]]; then
    echo "RunName is required in resume mode." >&2
    exit 2
  fi
  if [[ ! -f "$PROJECT_ROOT/experiment/all_runs/$RUN_NAME/optimizer.pt" ]]; then
    echo "Resume checkpoint not found: $PROJECT_ROOT/experiment/all_runs/$RUN_NAME" >&2
    exit 2
  fi
elif [[ -z "$RUN_NAME" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  if (( MAX_TRAIN_BATCHES == 0 )); then limit_tag=full; else limit_tag="${MAX_TRAIN_BATCHES}b"; fi
  RUN_NAME="sanity/server_x4_${MODE}_${limit_tag}_${stamp}"
fi

args=(
  main.py
  --dir_data "$DATA_ROOT"
  --model "$MODEL"
  --data_train DIV2K
  --data_test DIV2K
  --data_range "1-800/801-$VALIDATION_END"
  --scale 4
  --patch_size 256
  --batch_size 4
  --n_threads 8
  --ext img
  --epochs "$EPOCHS"
  --test_every 1000
  --lr "$LEARNING_RATE"
  --decay 200-400-600-800
  --gamma 0.5
  --loss "$LOSS_SPEC"
  --max_train_batches "$MAX_TRAIN_BATCHES"
  --print_every 10
)

if [[ "$MODEL" == "LFMNFeedback" ]]; then
  args+=(
    --feedback_stages "$FEEDBACK_STAGES"
    --feedback_mid "$FEEDBACK_MID"
    --feedback_lr_mult "$FEEDBACK_LR_MULT"
  )
fi
if [[ "$MODEL" == "LFMNFreq" ]]; then
  args+=(--freq_lr_mult "$FREQ_LR_MULT")
fi
if [[ "$MODEL" == "LFMNRDSM" || "$MODEL" == "LFMNRDSMDirect" ]]; then
  args+=(--rdsm_lr_mult "$RDSM_LR_MULT")
fi

case "$MODE" in
  scratch) args+=(--save "$RUN_NAME") ;;
  finetune) args+=(--pre_train "$PRETRAINED" --save "$RUN_NAME") ;;
  resume) args+=(--load "$RUN_NAME" --resume -1) ;;
esac

echo "Mode: $MODE; max batches/epoch: $MAX_TRAIN_BATCHES; target epochs: $EPOCHS"
echo "Model: $MODEL; feedback stages: $FEEDBACK_STAGES; feedback mid: $FEEDBACK_MID"
echo "Learning rate: $LEARNING_RATE"
echo "Feedback learning-rate multiplier: $FEEDBACK_LR_MULT"
echo "Frequency-prior learning-rate multiplier: $FREQ_LR_MULT"
echo "RDSM learning-rate multiplier: $RDSM_LR_MULT"
echo "Loss: $LOSS_SPEC"
echo "Validation: 0801-$VALIDATION_END"
echo "Output: experiment/all_runs/$RUN_NAME"

start_time="$(date +%s)"
cd "$LFMN_ROOT"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python "${args[@]}"
end_time="$(date +%s)"
elapsed="$((end_time - start_time))"
printf 'Wall-clock elapsed: %02d:%02d:%02d (%.1f minutes)\n' "$((elapsed/3600))" "$(((elapsed%3600)/60))" "$((elapsed%60))" "$(awk "BEGIN {print $elapsed/60}")"
