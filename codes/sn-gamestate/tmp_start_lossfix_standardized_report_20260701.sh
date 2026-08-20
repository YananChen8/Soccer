#!/usr/bin/env bash
set -euo pipefail
REPO=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
HUB=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_lossfix_l2mask
REPORT="$REPO/$HUB/report_standardized_20260701"
cd "$REPO"
mkdir -p "$REPORT"
nohup bash tmp_run_lossfix_standardized_report_20260701.sh 2 > "$REPORT/runner.log" 2>&1 &
echo $! > "$REPORT/runner.pid"
cat "$REPORT/runner.pid"
