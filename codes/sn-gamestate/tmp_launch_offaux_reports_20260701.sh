#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
ROOT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701
OUT=$ROOT/report_eval_visual_20260701
mkdir -p "$OUT/_logs"
runs=(fullft_offaux_last_motion_k3 fullft_offaux_last_nomotion_k3 fullft_offaux_stage1_motion_k3 fullft_offaux_stage1_nomotion_k3)
gpus=(0 1 2 3)
for i in "${!runs[@]}"; do
  r=${runs[$i]}
  g=${gpus[$i]}
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=plugins/calibration:. nohup "$PY" -u tmp_official_aux_report_eval_visual_20260701.py \
    --mode eval --split train --stride 100 --runs "$r" \
    --ckpt-root "$ROOT" --out-dir "$OUT/train_stride100_$r" \
    > "$OUT/_logs/train_stride100_$r.log" 2>&1 &
  echo $! > "$OUT/_logs/train_stride100_$r.pid"

  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=plugins/calibration:. nohup "$PY" -u tmp_official_aux_report_eval_visual_20260701.py \
    --mode all --split test --videos 116 117 118 119 120 121 122 123 --stride 100 --runs "$r" \
    --ckpt-root "$ROOT" --out-dir "$OUT/test_stride100_visual_$r" \
    > "$OUT/_logs/test_stride100_visual_$r.log" 2>&1 &
  echo $! > "$OUT/_logs/test_stride100_visual_$r.pid"
  sleep 2
done
