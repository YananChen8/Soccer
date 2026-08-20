#!/usr/bin/env bash
set -euo pipefail

cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
REPORT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701/report_eval_visual_20260701
LOGDIR=$REPORT/_logs/stride5_scatter_reproj25

echo PROCS
ps -ef | grep tmp_official_aux_report_eval_visual_20260701.py | grep 'stride 5' | grep -v grep || true
echo LOGS
for f in "$LOGDIR"/*.log; do
  [ -f "$f" ] || continue
  echo "== $f =="
  tail -20 "$f"
done
echo OUTPUTS
for d in "$REPORT"/test_stride5_scatter_reproj25_*; do
  [ -d "$d" ] || continue
  echo "== $d =="
  [ -f "$d/test_frame_scores.csv" ] && wc -l "$d/test_frame_scores.csv" || true
  find "$d/angle_reproj_scatter" -maxdepth 1 -type f 2>/dev/null | sort || true
done
