#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
ROOT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701/report_eval_visual_20260701
echo "PROCS"
ps -ef | grep 'tmp_official_aux_report_eval_visual_20260701.py --mode visual' | grep -v grep || true
echo "LOGS"
for f in "$ROOT"/_logs/revisualize_fixed_gt_points/*.log; do
  [ -f "$f" ] || continue
  echo "== $f =="
  tail -30 "$f"
done
echo "SAMPLE_SUMMARY"
for f in "$ROOT"/test_stride100_visual_*/visual_best_worst_k3/visual_summary.csv; do
  [ -f "$f" ] || continue
  echo "$f $(wc -l < "$f") lines"
done
