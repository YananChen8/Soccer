#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
LOG=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/lastpair_epoch_sweep_summary_20260702/summarizer.stdout.log
mkdir -p "$(dirname "$LOG")"
tmux kill-session -t summarize_epoch_sweep_tables 2>/dev/null || true
tmux new-session -d -s summarize_epoch_sweep_tables "$PY -u tmp_summarize_epoch_sweep_tables_20260702.py > $LOG 2>&1"
tmux ls | grep summarize_epoch_sweep_tables || true
