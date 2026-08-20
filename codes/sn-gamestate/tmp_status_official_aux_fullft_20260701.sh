#!/usr/bin/env bash
set -u
REPO=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
HUB=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701
ABS="$REPO/$HUB"
cd "$REPO" || exit 1
echo root "$ABS"
echo claims
find "$ABS/_watcher/claims" -mindepth 1 -maxdepth 2 -type f -name claim.txt -print -exec cat {} \; 2>/dev/null || true
echo watchers
find "$ABS/_watcher" -maxdepth 1 -name 'watch_*.log' -print -exec tail -8 {} \; 2>/dev/null || true
echo runs
for d in "$ABS"/fullft_offaux_*_k3; do
  [ -d "$d" ] || continue
  echo "== $(basename "$d") =="
  cat "$d/status.txt" 2>/dev/null || true
  tail -5 "$d/logs/train.log" 2>/dev/null || true
  if [ -f "$d/eval_test116_123_stride20/results.json" ]; then
    python - "$d/eval_test116_123_stride20/results.json" <<'PY'
import json, sys
p=sys.argv[1]
d=json.load(open(p))
print(json.dumps(d.get("aggregate",{}), indent=2))
PY
  fi
done
echo gpus
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
