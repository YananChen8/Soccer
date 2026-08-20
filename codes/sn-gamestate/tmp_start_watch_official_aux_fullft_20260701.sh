#!/usr/bin/env bash
set -euo pipefail
HOST_ID="${1:?host id required}"
MAX_SECONDS="${2:-36000}"
REPO=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
HUB=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701
WATCH="$REPO/$HUB/_watcher"
cd "$REPO"
mkdir -p "$WATCH"
nohup bash tmp_watch_official_aux_fullft_20260701.sh watch "$HOST_ID" "$MAX_SECONDS" > "$WATCH/nohup_watch_${HOST_ID}.log" 2>&1 &
echo $! > "$WATCH/watch_${HOST_ID}.pid"
cat "$WATCH/watch_${HOST_ID}.pid"
