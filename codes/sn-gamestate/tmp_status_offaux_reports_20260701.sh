#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
ROOT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701/report_eval_visual_20260701
echo "PROCS"
ps -ef | grep tmp_official_aux_report_eval_visual_20260701 | grep -v grep || true
echo "GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
echo "LOGS"
for f in "$ROOT"/_logs/*.log; do
  [ -f "$f" ] || continue
  echo "== $f =="
  tail -8 "$f" || true
done
echo "OUTPUTS"
find "$ROOT" -maxdepth 3 -type f \( -name '*metrics.md' -o -name '*results.json' -o -name '*frame_scores.csv' -o -name '*scatter*.png' -o -name 'visual_summary.csv' \) | sort | tail -80
