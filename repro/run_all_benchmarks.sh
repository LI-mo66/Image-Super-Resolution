#!/bin/bash
cd /e/fuxian_LFMN_jianghe/LFMN
PY=/e/anaconda/envs/dl/python.exe
DIR=E:/fuxian_LFMN_jianghe/datasets
OUT=/e/fuxian_LFMN_jianghe/repro/results_summary.txt

echo "LFMN 复现结果汇总（self_ensemble 开启，官方 benchmark 协议）" > "$OUT"
echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT"
echo "" >> "$OUT"

for scale in 2 3 4; do
  case $scale in
    2) w=996;;
    3) w=969;;
    4) w=939;;
  esac
  for ds in Set5 Set14 B100 Urban100; do
    echo ">>> 正在测 ${ds} x${scale} ..."
    line=$(PYTHONIOENCODING=utf-8 "$PY" main.py \
      --dir_data "$DIR" --model LFMN --data_test "$ds" \
      --scale "$scale" --pre_train "model/scale${scale}_model_${w}.pt" \
      --test_only --self_ensemble 2>&1 | grep -oE "\[[^]]*x${scale}\].*PSNR: [0-9.]+.*SSIM: [0-9.]+" | tail -1)
    echo "$line" | tee -a "$OUT"
  done
done

echo "" >> "$OUT"
echo "全部完成" >> "$OUT"
