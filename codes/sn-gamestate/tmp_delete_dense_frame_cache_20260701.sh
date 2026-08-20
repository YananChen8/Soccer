#!/usr/bin/env bash
set -euo pipefail

cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
p=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_frame_cache_20260701
manifest=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/deleted_dense_frame_cache_20260701_manifest.txt
if [ -d "$p" ]; then
  du -sh "$p" > "$manifest"
  rm -rf "$p"
  echo DELETED_DENSE_CACHE
  cat "$manifest"
else
  echo NO_DENSE_CACHE
fi
