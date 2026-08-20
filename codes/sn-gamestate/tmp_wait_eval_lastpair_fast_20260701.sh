#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate

PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
ROOT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k5_k15_motion_lastpair_fast_20260701
REPORT="$ROOT/report_eval_test_stride20"
RUNS=(
  fullft_cached_k15_stage1_motion_lastpair_fast_e5
  fullft_cached_k5_last_motion_lastpair_fast_e5
  fullft_cached_k5_stage1_motion_lastpair_fast_e5
)

echo "wait_eval_start $(date)"
for run in "${RUNS[@]}"; do
  pid_file="$ROOT/$run/train.pid"
  if [ -f "$pid_file" ]; then
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "waiting $run pid=$pid"
      wait "$pid" 2>/dev/null || while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    fi
  fi
done

mkdir -p "$REPORT"
echo "eval_start $(date)" | tee "$REPORT/eval.log"
env PYTHONPATH=plugins/calibration:. "$PY" -u tmp_official_aux_report_eval_visual_20260701.py \
  --mode eval \
  --split test \
  --videos 116 117 118 119 120 121 122 123 \
  --stride 20 \
  --runs baseline "${RUNS[@]}" \
  --ckpt-root "$ROOT" \
  --out-dir "$REPORT" \
  --device cuda \
  >> "$REPORT/eval.log" 2>&1
echo "eval_done $(date)" | tee -a "$REPORT/eval.log"
