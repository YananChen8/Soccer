"""Draw remaining BroadTrack-style overlays from existing frame_scores.csv."""
import argparse
import csv
import json
from pathlib import Path

import cv2

from broadtrack_min_ablation_round3 import image_path_for_frame
from visualize_broadtrack_ablation_round3 import draw_polyline_overlay


def delta(row, method, metric, sign=1.0):
    try:
        return sign * (float(row[f"{method}_{metric}"]) - float(row[f"baseline_{metric}"]))
    except Exception:
        return -1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--vis-dir", required=True)
    ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    vis_dir = Path(args.vis_dir)
    frame_dir = vis_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(vis_dir / "frame_scores.csv", newline="")))
    params = json.load(open(run_dir / "eval" / "params.json"))

    made = []
    for method in ["radial_k1", "tripod"]:
        top = sorted(rows, key=lambda r: delta(r, method, "reproj", sign=-1.0), reverse=True)[: args.topk]
        for rank, row in enumerate(top, 1):
            vid = row["video"]
            gid = row["gid"]
            frame = int(gid) % 1000000 + int(vid) * 1000000 + 3000000000
            img = cv2.imread(image_path_for_frame(vid, frame))
            if img is None:
                continue
            out = frame_dir / f"{method}_top{rank}_{vid}_{gid}.jpg"
            title = (
                f"SNGS-{vid} {gid} {method}: "
                f"dPoint={delta(row, method, 'point'):+.3f} "
                f"dLine={delta(row, method, 'line'):+.3f} "
                f"dReproj={-delta(row, method, 'reproj', sign=-1.0):+.2f}"
            )
            draw_polyline_overlay(
                img,
                {
                    "baseline": params["baseline"].get(vid, {}).get(gid, {}),
                    method: params[method].get(vid, {}).get(gid, {}),
                },
                out,
                title,
            )
            made.append(str(out))

    summary = [
        "# BroadTrack-style Visualization Summary",
        "",
        f"Run dir: `{run_dir}`",
        f"Frame score rows: {len(rows)}",
        "",
        "## Statistics",
        "- `frame_scores.csv`: per-frame metrics for sampled frames.",
        "- `summary_delta_point.png`: point delta distribution.",
        "- `summary_delta_line.png`: line delta distribution.",
        "- `summary_delta_reproj.png`: reprojection improvement distribution.",
        "",
        "## Frame Overlays",
        "- `frames/flow_top*.jpg`: flow outlier repair cases.",
        "- `frames/radial_k1_top*.jpg`: baseline vs radial projection overlays.",
        "- `frames/tripod_top*.jpg`: baseline vs tripod projection overlays.",
        "",
        "Generated overlay files:",
    ]
    summary += [f"- `{p}`" for p in made]
    (vis_dir / "VIS_SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(vis_dir / "VIS_SUMMARY.md")


if __name__ == "__main__":
    main()
