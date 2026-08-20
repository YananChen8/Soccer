#!/usr/bin/env bash
set -euo pipefail
REPO=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
ROOT="$REPO/outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_cached_speed_20260701"
REPORT="$ROOT/report"
cd "$REPO"
mkdir -p "$REPORT"
nohup bash tmp_run_cached_speed_pipeline_20260701.sh 2 > "$REPORT/pipeline.log" 2>&1 &
echo $! > "$REPORT/pipeline.pid"
cat "$REPORT/pipeline.pid"
