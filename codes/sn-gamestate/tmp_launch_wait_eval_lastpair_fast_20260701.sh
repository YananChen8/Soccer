#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate

PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
ROOT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k5_k15_motion_lastpair_fast_20260701
mkdir -p "$ROOT"

nohup env PYTHONPATH=plugins/calibration:. "$PY" -u tmp_wait_eval_lastpair_fast_20260701.py \
  > "$ROOT/wait_eval_stdout.log" 2>&1 &
echo $! > "$ROOT/wait_eval.pid"
echo "started wait_eval pid=$(cat "$ROOT/wait_eval.pid")"
