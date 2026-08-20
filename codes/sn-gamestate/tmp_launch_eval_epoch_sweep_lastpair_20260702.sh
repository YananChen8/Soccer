#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
LOG=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/eval_epoch_sweep_lastpair_20260702.stdout.log
tmux kill-session -t eval_epoch_sweep_lastpair 2>/dev/null || true
tmux new-session -d -s eval_epoch_sweep_lastpair "$PY -u tmp_eval_epoch_sweep_lastpair_20260702.py > $LOG 2>&1"
tmux ls | grep eval_epoch_sweep_lastpair || true
