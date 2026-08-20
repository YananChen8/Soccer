#!/usr/bin/env bash
set -euo pipefail

cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
ROOT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_memmap_cache_20260701
echo PROCS
ps -ef | grep tmp_build_temporal_memmap_cache_20260701.py | grep -v grep || true
echo DISK
df -h /remote-home
echo SIZE
du -sh "$ROOT" 2>/dev/null || true
echo PROGRESS
for f in "$ROOT"/train/progress.json "$ROOT"/test/progress.json; do
  [ -f "$f" ] || continue
  echo "== $f =="
  cat "$f"
done
echo LOGS
for f in "$ROOT"/logs/cache_train.log "$ROOT"/logs/cache_test.log; do
  [ -f "$f" ] || continue
  echo "== $f =="
  tail -20 "$f"
done
echo FILES
find "$ROOT" -maxdepth 2 -type f \( -name 'cache_meta.json' -o -name 'compact_labels.npz' -o -name 'images_u8_chw_540x960.dat' -o -name 'errors.json' \) -exec ls -lh {} \; 2>/dev/null || true
