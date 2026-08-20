#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
ROOT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701
REPORT=$ROOT/report_eval_visual_20260701
LOG=$REPORT/_logs/revisualize_fixed_gt_points
mkdir -p "$LOG"
runs=(fullft_offaux_last_motion_k3 fullft_offaux_last_nomotion_k3 fullft_offaux_stage1_motion_k3 fullft_offaux_stage1_nomotion_k3)
gpus=(0 1 2 3)
for i in "${!runs[@]}"; do
  r=${runs[$i]}
  g=${gpus[$i]}
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=plugins/calibration:. nohup "$PY" -u tmp_official_aux_report_eval_visual_20260701.py \
    --mode visual --split test --runs "$r" \
    --ckpt-root "$ROOT" \
    --out-dir "$REPORT/test_stride100_visual_$r" \
    > "$LOG/$r.log" 2>&1 &
  echo $! > "$LOG/$r.pid"
  sleep 2
done
echo "STARTED fixed GT point redraw"
ls -l "$LOG"/*.pid
