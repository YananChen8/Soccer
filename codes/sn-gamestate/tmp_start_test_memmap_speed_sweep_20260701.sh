#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
OUT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701/test_memmap_speed_sweep_20260701
CACHE=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_memmap_cache_20260701/test
mkdir -p "$OUT"
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=plugins/calibration:. "$PY" -u tmp_test_memmap_temporal_speed_sweep_20260701.py \
  --cache-dir "$CACHE" \
  --out-dir "$OUT" \
  --steps 20 \
  --max-samples 4096 \
  --batches 4,8,12,16 \
  --workers 0,4,8 \
  > "$OUT/sweep.log" 2>&1 &
echo $! > "$OUT/sweep.pid"
echo "started pid=$(cat "$OUT/sweep.pid") out=$OUT"
