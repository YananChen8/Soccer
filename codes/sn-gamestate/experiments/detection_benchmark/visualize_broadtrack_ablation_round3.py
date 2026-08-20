"""Visualize minimal BroadTrack-style calibration ablations.

Outputs:
- frame_scores.csv with per-frame method metrics and deltas
- summary PNGs
- selected frame overlays for flow, radial_k1, and tripod
"""
import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from broadtrack_min_ablation_round3 import (
    DATA_ROOT,
    decode_keypoints,
    flow_outlier_replace,
    image_path_for_frame,
    ransac_inlier_keys,
)
from cached_full_test_round2 import smoothness
from sn_gamestate.structured_calibration.metrics import (
    HEIGHT,
    THRESHOLD,
    WIDTH,
    evaluate_camera_prediction,
    get_polylines,
    load_gt_lines_for_video,
    mirror_labels,
)


def frame_metric(params, gt_lines):
    if not isinstance(params, dict) or not params:
        return None
    try:
        pred = get_polylines(params, WIDTH, HEIGHT, sampling_factor=0.9)
    except Exception:
        return None

    def one(gl):
        c, _, r = evaluate_camera_prediction(pred, gl, THRESHOLD)
        point = c[0, 0] / c.sum() if c.sum() > 0 else 0.0
        per_line = [np.mean([e <= THRESHOLD for e in es]) for es in r.values() if es]
        line = float(np.mean(per_line)) if per_line else 0.0
        reproj = [e for es in r.values() for e in es]
        return point, line, float(np.mean(reproj)) if reproj else np.nan

    m1 = one(gt_lines)
    try:
        m2 = one(mirror_labels(gt_lines))
    except Exception:
        m2 = None
    return m1 if (m2 is None or m1[0] >= m2[0]) else m2


def draw_polyline_overlay(image, params_by_name, out_path, title):
    colors = {
        "baseline": (255, 80, 80),
        "flow": (80, 210, 255),
        "radial_k1": (80, 255, 120),
        "tripod": (255, 180, 60),
    }
    img = image.copy()
    for name, params in params_by_name.items():
        if not params:
            continue
        try:
            polys = get_polylines(params, WIDTH, HEIGHT, sampling_factor=0.9)
        except Exception:
            continue
        color = colors.get(name, (240, 240, 240))
        for pts in polys.values():
            clean = []
            for p in pts:
                if isinstance(p, dict):
                    x, y = p.get("x"), p.get("y")
                else:
                    x, y = p[0], p[1]
                if x is None or y is None:
                    continue
                x, y = float(x), float(y)
                if -2.0 <= x <= 2.0 and -2.0 <= y <= 2.0:
                    x, y = x * WIDTH, y * HEIGHT
                clean.append([x, y])
            arr = np.array(clean, dtype=np.int32).reshape(-1, 1, 2)
            if len(arr) >= 2:
                cv2.polylines(img, [arr], False, color, 2, cv2.LINE_AA)
    cv2.putText(img, title, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)


def load_cache(cache_root, vid, frame):
    files = sorted(glob.glob(f"{cache_root}/test/SNGS-{vid}/frame_*.npz"))
    for path in files:
        with np.load(path) as d:
            if int(d["frame"]) == int(frame):
                return path, d["kp_hm"].copy(), d["line_hm"].copy()
    return None, None, None


def draw_flow_case(cache_root, vid, gid, stride, out_path):
    frame = int(gid) % 1000000 + int(vid) * 1000000 + 3000000000
    prev_frame = frame - stride
    cur_path, kp_hm, line_hm = load_cache(cache_root, vid, frame)
    prev_path, prev_kp_hm, prev_line_hm = load_cache(cache_root, vid, prev_frame)
    if cur_path is None or prev_path is None:
        return False
    cur_img = cv2.imread(image_path_for_frame(vid, frame))
    prev_img = cv2.imread(image_path_for_frame(vid, prev_frame))
    if cur_img is None or prev_img is None:
        return False
    base = decode_keypoints(kp_hm, line_hm, "cpu")
    prev = decode_keypoints(prev_kp_hm, prev_line_hm, "cpu")
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    stats = defaultdict(int)
    repaired = flow_outlier_replace(base, prev_gray, cur_gray, prev, stats)
    outliers = sorted(set(base) - ransac_inlier_keys(base))

    canvas = np.concatenate([prev_img, cur_img], axis=1)
    for key in outliers:
        if key not in base:
            continue
        bx, by = int(base[key]["x"] * 1920), int(base[key]["y"] * 1080)
        cv2.drawMarker(canvas, (1920 + bx, by), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
        if key in prev:
            px, py = int(prev[key]["x"] * 1920), int(prev[key]["y"] * 1080)
            cv2.circle(canvas, (px, py), 6, (255, 200, 60), 2)
        if key in repaired:
            rx, ry = int(repaired[key]["x"] * 1920), int(repaired[key]["y"] * 1080)
            cv2.circle(canvas, (1920 + rx, ry), 7, (80, 255, 120), 2)
            cv2.arrowedLine(canvas, (1920 + bx, by), (1920 + rx, ry), (80, 255, 120), 2, tipLength=0.2)
            cv2.putText(canvas, str(key), (1920 + rx + 8, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 120), 2)
    cv2.putText(canvas, f"SNGS-{vid} frame {frame} flow: red=HRNet outlier, green=LK repair",
                (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)
    return True


def plot_summary(rows, out_dir):
    methods = ["flow", "radial_k1", "tripod"]
    metrics = ["point", "line", "reproj"]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        data = []
        labels = []
        for m in methods:
            vals = [float(r[f"{m}_{metric}"]) - float(r[f"baseline_{metric}"])
                    for r in rows if r.get(f"{m}_{metric}") not in ("", "nan") and r.get(f"baseline_{metric}") not in ("", "nan")]
            if metric == "reproj":
                vals = [-v for v in vals]
            data.append(vals)
            labels.append(m)
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel(f"{metric} improvement vs baseline")
        ax.set_title(f"BroadTrack-style {metric} deltas")
        fig.tight_layout()
        fig.savefig(out_dir / f"summary_delta_{metric}.png", dpi=160)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-rows", type=int, default=0, help="Max scored frames to evaluate for visual selection; 0 means all.")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)

    params = json.load(open(run_dir / "eval" / "params.json"))
    vids = sorted({vid for method in params.values() for vid in method})
    gt = {vid: load_gt_lines_for_video(DATA_ROOT, vid) for vid in vids}
    rows = []
    for vid in vids:
        gids = sorted(set().union(*[set(params.get(m, {}).get(vid, {})) for m in params]))
        for gid in gids:
            if gid not in gt[vid]:
                continue
            row = {"video": vid, "gid": gid}
            for method in params:
                metric = frame_metric(params[method].get(vid, {}).get(gid, {}), gt[vid][gid])
                if metric is None:
                    row[f"{method}_point"] = ""
                    row[f"{method}_line"] = ""
                    row[f"{method}_reproj"] = ""
                else:
                    row[f"{method}_point"], row[f"{method}_line"], row[f"{method}_reproj"] = metric
            rows.append(row)
            if args.max_rows and len(rows) >= args.max_rows:
                break
        if args.max_rows and len(rows) >= args.max_rows:
            break

    fieldnames = ["video", "gid"]
    for method in sorted(params):
        fieldnames += [f"{method}_point", f"{method}_line", f"{method}_reproj"]
    with open(out_dir / "frame_scores.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)
    plot_summary(rows, out_dir)

    def valid_delta(row, method, metric, sign=1):
        try:
            return sign * (float(row[f"{method}_{metric}"]) - float(row[f"baseline_{metric}"]))
        except Exception:
            return -999999.0

    # Flow keypoint repair cases.
    for rank, row in enumerate(sorted(rows, key=lambda r: valid_delta(r, "flow", "point"), reverse=True)[: args.topk], 1):
        draw_flow_case(args.cache_root, row["video"], row["gid"], args.stride, out_dir / "frames" / f"flow_top{rank}_{row['video']}_{row['gid']}.jpg")

    # Projection overlays for radial and tripod.
    for method, metric, sign in [("radial_k1", "reproj", -1), ("tripod", "reproj", -1)]:
        top = sorted(rows, key=lambda r: valid_delta(r, method, metric, sign=sign), reverse=True)[: args.topk]
        for rank, row in enumerate(top, 1):
            vid = row["video"]
            frame = int(row["gid"]) % 1000000 + int(vid) * 1000000 + 3000000000
            img = cv2.imread(image_path_for_frame(vid, frame))
            if img is None:
                continue
            draw_polyline_overlay(
                img,
                {
                    "baseline": params["baseline"].get(vid, {}).get(row["gid"], {}),
                    method: params[method].get(vid, {}).get(row["gid"], {}),
                },
                out_dir / "frames" / f"{method}_top{rank}_{vid}_{row['gid']}.jpg",
                f"SNGS-{vid} {row['gid']} baseline(red) vs {method}",
            )

    md = [
        "# BroadTrack-style Visualization Summary",
        "",
        f"Run dir: `{run_dir}`",
        f"Rows: {len(rows)}",
        "",
        "## Files",
        "- `frame_scores.csv`: per-frame metrics for each method.",
        "- `summary_delta_point.png`, `summary_delta_line.png`, `summary_delta_reproj.png`: method delta distributions.",
        "- `frames/*.jpg`: selected visual cases.",
    ]
    (out_dir / "VIS_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out_dir / "VIS_SUMMARY.md")


if __name__ == "__main__":
    main()
