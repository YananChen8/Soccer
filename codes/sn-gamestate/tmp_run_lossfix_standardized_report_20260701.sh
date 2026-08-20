#!/usr/bin/env bash
set -euo pipefail

REPO=/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
HUB=outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_lossfix_l2mask
ABS_HUB="$REPO/$HUB"
REPORT="$ABS_HUB/report_standardized_20260701"
GPU="${1:-2}"

cd "$REPO"
mkdir -p "$REPORT"

"$PY" - <<'PY'
import csv
from pathlib import Path
src = Path("outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_lossfix_l2mask/visual_points_lines_20260630/summary.csv")
out = Path("outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_lossfix_l2mask/report_standardized_20260701/standard_viz_frames_20260701.csv")
seen, rows = set(), []
with src.open(newline="") as f:
    for r in csv.DictReader(f):
        key = (str(r["video"]), str(r["frame"]).zfill(6))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"video": key[0], "frame": key[1]})
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["video", "frame"])
    w.writeheader()
    w.writerows(rows)
print("standard_viz_frames", len(rows), out, flush=True)
PY

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=plugins/calibration:. "$PY" -u tmp_summarize_lossfix_training_curves_20260701.py \
  --root "$ABS_HUB" \
  --out-dir "$REPORT/loss_curves" \
  > "$REPORT/loss_curves.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=plugins/calibration:. "$PY" -u tmp_visualize_lossfix_models_gt_metrics_20260701.py \
  --frame-csv "$REPORT/standard_viz_frames_20260701.csv" \
  --ckpt-root "$ABS_HUB" \
  --results-root "$ABS_HUB" \
  --out-dir "$ABS_HUB/visual_points_lines_gt_metrics_20260701" \
  > "$REPORT/visual_gt_metrics.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=plugins/calibration:. "$PY" -u tmp_eval_temporal_feature_fusion_train_manifest_20260701.py \
  --stride 100 \
  --ckpt-root "$ABS_HUB" \
  --out-dir "$ABS_HUB/eval_train_manifest_stride100_20260701" \
  > "$REPORT/eval_train_manifest_stride100.log" 2>&1

echo "DONE $(date '+%F %T')" > "$REPORT/DONE"
