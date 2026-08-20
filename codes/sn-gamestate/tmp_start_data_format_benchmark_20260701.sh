#!/usr/bin/env bash
set -euo pipefail

cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
OUT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701/data_format_benchmark_20260701
mkdir -p "$OUT"
nohup env CUDA_VISIBLE_DEVICES=4 "$PY" -u tmp_benchmark_temporal_data_formats_20260701.py \
  --max-frames 1024 \
  --max-videos 4 \
  --steps 80 \
  --batch-size 8 \
  --device cuda \
  > "$OUT/benchmark.log" 2>&1 &
echo "$!" > "$OUT/benchmark.pid"
echo "launched data format benchmark pid=$(cat "$OUT/benchmark.pid") out=$OUT"
