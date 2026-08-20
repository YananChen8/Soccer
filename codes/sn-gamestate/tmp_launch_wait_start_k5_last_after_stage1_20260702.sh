#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
tmux kill-session -t wait_start_k5_last_after_stage1 2>/dev/null || true
tmux new-session -d -s wait_start_k5_last_after_stage1 "$PY -u tmp_wait_start_k5_last_after_stage1_20260702.py"
tmux ls | grep wait_start_k5_last_after_stage1 || true
