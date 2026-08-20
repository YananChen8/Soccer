#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
HUB=outputs/gsr/temporal_hrnet/temporal_calib_results_hub
OLD="$HUB/full_finetune_temporal_nbjw_k3_cached_speed_20260701/cache_train_mfpv120_k3_u8"
ROOT="$HUB/full_finetune_temporal_nbjw_k3_frame_cache_20260701"
LOG="$ROOT/logs"
mkdir -p "$LOG"

case "$OLD" in
  outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_cached_speed_20260701/cache_train_mfpv120_k3_u8) ;;
  *) echo "refuse unexpected OLD=$OLD"; exit 2 ;;
esac
if [ -d "$OLD" ]; then
  du -sh "$OLD" > "$LOG/deleted_window_cache_manifest.txt" || true
  find "$OLD" -maxdepth 1 -type f | head -20 >> "$LOG/deleted_window_cache_manifest.txt" || true
  rm -rf "$OLD"
  echo "DELETED $OLD $(date '+%F %T')" >> "$LOG/deleted_window_cache_manifest.txt"
fi

df -h "$HUB" > "$LOG/disk_before_cache.txt"

nohup "$PY" -u tmp_cache_offaux_frame_tensors_20260701.py \
  --split train \
  --out-dir "$ROOT/train_frame_cache_u8" \
  --shard-size 32 \
  > "$LOG/cache_train.log" 2>&1 &
echo $! > "$LOG/cache_train.pid"

nohup "$PY" -u tmp_cache_offaux_frame_tensors_20260701.py \
  --split test \
  --out-dir "$ROOT/test_frame_cache_u8" \
  --shard-size 32 \
  > "$LOG/cache_test.log" 2>&1 &
echo $! > "$LOG/cache_test.pid"

cat > "$ROOT/README.md" <<'EOF'
# Frame-Level Temporal HRNet Cache

This cache stores each frame once. Temporal windows should be assembled by the Dataset at training time from `frames_manifest.csv`, so K can be changed without rebuilding cache.

Shard payload format:
- `image_u8`: `[N,3,540,960]`
- `gt_u8`: `[N,58,270,480]`, divide by 255 during training
- `mask_u8`: `[N,58]`
- metadata lists: `video`, `image_id`, `frame_index`, `dst_image`, `dst_json`

Use `frames_manifest.csv` to map `(video, frame_index)` to `(shard_index, offset)`.
EOF

echo "STARTED frame cache root=$ROOT"
cat "$LOG/cache_train.pid"
cat "$LOG/cache_test.pid"
