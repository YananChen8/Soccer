#!/usr/bin/env bash
set -euo pipefail
REPO=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
ROOT="$REPO/outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_cached_speed_20260701"
REPORT="$ROOT/report"
GPU="${1:-2}"
cd "$REPO"
mkdir -p "$REPORT"
echo "FINALIZER_START $(date '+%F %T')" > "$REPORT/finalizer.log"
for i in $(seq 1 720); do
  if [ -f "$ROOT/speed_sweep_last/speed_sweep_results.json" ]; then
    break
  fi
  echo "$(date '+%F %T') waiting speed_sweep_results.json i=$i" >> "$REPORT/finalizer.log"
  sleep 60
done
if [ ! -f "$ROOT/speed_sweep_last/speed_sweep_results.json" ]; then
  echo "TIMEOUT waiting sweep results" >> "$REPORT/finalizer.log"
  exit 4
fi
"$PY" -u tmp_summarize_cached_speed_recommendation_20260701.py \
  --sweep-json "$ROOT/speed_sweep_last/speed_sweep_results.json" \
  --out-dir "$ROOT/speed_sweep_last" \
  >> "$REPORT/finalizer.log" 2>&1

# Baseline fast proxy: quick test-side loader/metric benchmark without camera solve.
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=plugins/calibration:. "$PY" -u tmp_fast_eval_peak_proxy_20260701.py \
  --videos 116 117 118 119 120 121 122 123 \
  --stride 20 \
  --batch-size 16 \
  --workers 4 \
  --out-dir "$ROOT/fast_eval_proxy_baseline_stride20" \
  >> "$REPORT/finalizer.log" 2>&1 || true

echo "FINALIZER_DONE $(date '+%F %T')" >> "$REPORT/finalizer.log"
