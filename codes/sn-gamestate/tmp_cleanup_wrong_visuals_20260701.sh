#!/usr/bin/env bash
set -euo pipefail

cd /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate

HUB=outputs/gsr/temporal_hrnet/temporal_calib_results_hub
MANIFEST="$HUB/cleanup_wrong_visuals_20260701_manifest.txt"
: > "$MANIFEST"

add_if_exists() {
  local p="$1"
  if [ -e "$p" ]; then
    printf '%s\n' "$p" >> "$MANIFEST"
  fi
}

# Wrong folded-angle scatter dirs from the earlier official-aux visual pass.
find "$HUB/full_finetune_temporal_nbjw_k3_official_aux_20260701/report_eval_visual_20260701" \
  -path '*/angle_reproj_scatter' -type d -print >> "$MANIFEST" 2>/dev/null || true

# Aborted stride=5 folded-angle outputs and smoke outputs.
find "$HUB/full_finetune_temporal_nbjw_k3_official_aux_20260701/report_eval_visual_20260701" \
  -maxdepth 1 -type d \( -name 'test_stride5_scatter_reproj25_*' -o -name '_smoke_stride5_scatter*' \) \
  -print >> "$MANIFEST" 2>/dev/null || true

# Older lossfix visualizations were produced before the fixed GT/keypoint drawing flow.
add_if_exists "$HUB/full_finetune_temporal_nbjw_k3_lossfix_l2mask/visual_points_lines_20260630"
add_if_exists "$HUB/full_finetune_temporal_nbjw_k3_lossfix_l2mask/visual_points_lines_gt_metrics_20260701"

sort -u "$MANIFEST" -o "$MANIFEST"
echo "MANIFEST $MANIFEST"
cat "$MANIFEST"

if [ "${DO_DELETE:-0}" = "1" ]; then
  while IFS= read -r p; do
    [ -n "$p" ] || continue
    rm -rf -- "$p"
  done < "$MANIFEST"
  echo "DELETED"
fi
