#!/usr/bin/env bash
set -u
REPO=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
ROOT="$REPO/outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_cached_speed_20260701"
REPORT="$ROOT/report"
echo root "$ROOT"
echo pid
cat "$REPORT/pipeline.pid" 2>/dev/null || true
if [ -f "$REPORT/pipeline.pid" ]; then
  ps -p "$(cat "$REPORT/pipeline.pid")" -o pid,etime,cmd || true
fi
echo status
cat "$REPORT/status.txt" 2>/dev/null || true
echo cache_tail
tail -20 "$REPORT/cache.log" 2>/dev/null || true
echo sweep_tail
tail -20 "$REPORT/speed_sweep_last.log" 2>/dev/null || true
echo outputs
find "$ROOT" -maxdepth 3 -type f 2>/dev/null | sort | tail -40
echo disk
df -h "$ROOT" 2>/dev/null || df -h "$REPO"
