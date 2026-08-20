#!/usr/bin/env bash
set -u
REPO=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
HUB=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_lossfix_l2mask
REPORT="$REPO/$HUB/report_standardized_20260701"
cd "$REPO" || exit 1
echo runner_pid
cat "$REPORT/runner.pid" 2>/dev/null || true
echo ps
if [ -f "$REPORT/runner.pid" ]; then
  ps -p "$(cat "$REPORT/runner.pid")" -o pid,etime,cmd || true
fi
echo runner_log
tail -30 "$REPORT/runner.log" 2>/dev/null || true
echo report_files
find "$REPORT" -maxdepth 2 -type f 2>/dev/null | sort | tail -30
echo visual_count
find "$REPO/$HUB/visual_points_lines_gt_metrics_20260701" -name '*.png' 2>/dev/null | wc -l
echo train_eval_files
find "$REPO/$HUB/eval_train_manifest_stride100_20260701" -maxdepth 3 -type f 2>/dev/null | sort | tail -20
echo gpu2
nvidia-smi -i 2 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
