#!/usr/bin/env bash
set -euo pipefail

cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
ROOT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_memmap_cache_20260701
mkdir -p "$ROOT/logs"

for split in train test; do
  nohup "$PY" -u tmp_build_temporal_memmap_cache_20260701.py \
    --split "$split" \
    --out-root "$ROOT" \
    --flush-every 500 \
    > "$ROOT/logs/cache_${split}.log" 2>&1 &
  echo "$!" > "$ROOT/logs/cache_${split}.pid"
  echo "launched split=$split pid=$(cat "$ROOT/logs/cache_${split}.pid") out=$ROOT/$split"
done
