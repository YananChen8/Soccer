#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
OUT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701/report_eval_visual_20260701
tar -czf "$OUT/test_stride100_visual_dirs_20260701.tar.gz" -C "$OUT" \
  test_stride100_visual_fullft_offaux_last_motion_k3 \
  test_stride100_visual_fullft_offaux_last_nomotion_k3 \
  test_stride100_visual_fullft_offaux_stage1_motion_k3 \
  test_stride100_visual_fullft_offaux_stage1_nomotion_k3
ls -lh "$OUT/test_stride100_visual_dirs_20260701.tar.gz"
