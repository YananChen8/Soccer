#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
ROOT=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_frame_cache_20260701
echo "PROCS"
ps -ef | grep tmp_cache_offaux_frame_tensors_20260701 | grep -v grep || true
echo "DISK"
df -h outputs/gsr/temporal_hrnet/temporal_calib_results_hub
echo "SIZES"
du -sh "$ROOT" "$ROOT"/train_frame_cache_u8 "$ROOT"/test_frame_cache_u8 2>/dev/null || true
echo "TRAIN_LOG"
tail -8 "$ROOT/logs/cache_train.log" 2>/dev/null || true
echo "TEST_LOG"
tail -8 "$ROOT/logs/cache_test.log" 2>/dev/null || true
echo "MANIFESTS"
ls -lh "$ROOT"/train_frame_cache_u8/cache_manifest.json "$ROOT"/test_frame_cache_u8/cache_manifest.json 2>/dev/null || true
