#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
ROOT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701
OUT=$ROOT/report_eval_visual_20260701
mkdir -p "$OUT/_logs"
runs=(fullft_offaux_last_motion_k3 fullft_offaux_last_nomotion_k3 fullft_offaux_stage1_motion_k3 fullft_offaux_stage1_nomotion_k3)
gpus=(1 3 4 5)
for i in "${!runs[@]}"; do
  r=${runs[$i]}
  g=${gpus[$i]}
  out="$OUT/train_stride20_fixed_$r"
  log="$OUT/_logs/train_stride20_fixed_$r.log"
  pidf="$OUT/_logs/train_stride20_fixed_$r.pid"
  if [ -f "$out/train_results.json" ]; then
    echo "skip existing $r"
    continue
  fi
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=plugins/calibration:. nohup "$PY" -u tmp_official_aux_report_eval_visual_20260701.py \
    --mode eval --split train --stride 20 --runs "$r" \
    --ckpt-root "$ROOT" --out-dir "$out" \
    > "$log" 2>&1 &
  echo $! > "$pidf"
  echo "launched $r gpu=$g pid=$(cat "$pidf") out=$out"
  sleep 2
done
