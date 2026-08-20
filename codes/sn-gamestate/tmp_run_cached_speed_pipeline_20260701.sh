#!/usr/bin/env bash
set -euo pipefail

REPO=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
ROOT="$REPO/outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_cached_speed_20260701"
CACHE="$ROOT/cache_train_mfpv120_k3_u8"
REPORT="$ROOT/report"
GPU="${1:-2}"

mkdir -p "$ROOT" "$REPORT"
cd "$REPO"

avail_kb=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
required_kb=$((180 * 1024 * 1024))
{
  echo "START $(date '+%F %T')"
  echo "root=$ROOT"
  echo "cache=$CACHE"
  echo "gpu=$GPU"
  echo "avail_kb=$avail_kb required_kb=$required_kb"
} > "$REPORT/status.txt"
if [ "$avail_kb" -lt "$required_kb" ]; then
  echo "FAILED_SPACE avail_kb=$avail_kb required_kb=$required_kb" >> "$REPORT/status.txt"
  exit 3
fi

if [ ! -f "$CACHE/cache_manifest.json" ]; then
  ionice -c2 -n7 nice -n 10 "$PY" -u tmp_cache_offaux_train_tensors_20260701.py \
    --out-dir "$CACHE" \
    --window-size 3 \
    --shard-size 32 \
    --max-frames-per-video 120 \
    > "$REPORT/cache.log" 2>&1
fi
echo "CACHE_DONE $(date '+%F %T')" >> "$REPORT/status.txt"

# Wait until requested GPU has no compute process for 3 minutes.
free_since=0
while true; do
  now=$(date +%s)
  apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '^[[:space:]]*[0-9]+' | wc -l)
  util=$(nvidia-smi -i "$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
  echo "$(date '+%F %T') gpu=$GPU util=${util:-NA} apps=$apps free_since=$free_since" >> "$REPORT/wait_gpu.log"
  if [ "${util:-999}" = "0" ] && [ "$apps" = "0" ]; then
    if [ "$free_since" = "0" ]; then
      free_since=$now
    fi
    if [ $((now - free_since)) -ge 180 ]; then
      break
    fi
  else
    free_since=0
  fi
  sleep 30
done

echo "SWEEP_START $(date '+%F %T')" >> "$REPORT/status.txt"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=plugins/calibration:. "$PY" -u tmp_train_offaux_cached_speed_sweep_20260701.py \
  --cache-dir "$CACHE" \
  --out-dir "$ROOT/speed_sweep_last" \
  --fusion-level last \
  --residual-scale 0.05 \
  --max-steps 300 \
  --grad-accum-steps 1 \
  --batches 4,8,12 \
  --workers 0,4,8 \
  > "$REPORT/speed_sweep_last.log" 2>&1
"$PY" -u tmp_summarize_cached_speed_recommendation_20260701.py \
  --sweep-json "$ROOT/speed_sweep_last/speed_sweep_results.json" \
  --out-dir "$ROOT/speed_sweep_last" \
  > "$REPORT/recommendation.log" 2>&1
echo "DONE $(date '+%F %T')" >> "$REPORT/status.txt"
