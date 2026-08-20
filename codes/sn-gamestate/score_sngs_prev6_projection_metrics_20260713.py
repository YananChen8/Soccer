import csv
import json
import math
from pathlib import Path

import numpy as np

from sn_gamestate.structured_calibration.metrics import get_polylines


ROOT = Path("/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate")
OUT = ROOT / "outputs/paper_repro/high_accuracy_calib_20260702"
PREV6 = OUT / "sngs_probe_eval_prev6_20260713"
TEMPLATE_DB = OUT / "nbjw_oracle_db_train_full"
PROBES = [
    "train_oracle_rendered_q300_db10000",
    "test116123_oracle_rendered_q300_db10000",
]
RUNS = [("reg", "10k"), ("feat", "10k"), ("reg", "50k"), ("feat", "50k"), ("reg", "100k"), ("feat", "100k")]
THRESHOLDS = [5, 10, 15, 20]


def load_template():
    with (TEMPLATE_DB / "items.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)["params"]
    raise RuntimeError(f"empty template db: {TEMPLATE_DB}")


def make_cam(row, template, width=960, height=540, src_width=1920, src_height=1080):
    params = dict(template)
    sx = width / float(src_width)
    sy = height / float(src_height)
    params["pan_degrees"] = float(row["pan"])
    params["roll_degrees"] = float(row["roll"])
    params["tilt_degrees"] = float(row["tilt"])
    params["x_focal_length"] = float(row["focal"]) * sx
    params["y_focal_length"] = float(row["focal"]) * sy
    params["principal_point"] = [width / 2.0, height / 2.0]
    return params


def xy(p):
    return float(p["x"]), float(p["y"])


def visible(points, width, height, pad=2):
    out = []
    for p in points:
        x, y = xy(p)
        if -pad <= x < width + pad and -pad <= y < height + pad:
            out.append((x, y))
    return out


def interp_polyline(points, step=20.0):
    if len(points) < 2:
        return []
    arr = []
    for a, b in zip(points[:-1], points[1:]):
        ax, ay = a
        bx, by = b
        dist = math.hypot(bx - ax, by - ay)
        n = max(1, int(dist / step))
        for i in range(n):
            t = i / n
            arr.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    arr.append(points[-1])
    return arr


def render_geom(row, template, width=960, height=540):
    polylines = get_polylines(make_cam(row, template, width, height), width, height, sampling_factor=0.9)
    line_points = []
    key_points = []
    for _name, line in polylines.items():
        pts = visible(line, width, height)
        if len(pts) < 2:
            continue
        line_points.extend(interp_polyline(pts, step=20.0))
        key_points.append(pts[0])
        key_points.append(pts[-1])
        key_points.append(pts[len(pts) // 2])
    line_points = np.asarray(line_points, dtype=np.float32)
    if len(line_points) > 1200:
        take = np.linspace(0, len(line_points) - 1, 1200, dtype=np.int64)
        line_points = line_points[take]
    return line_points, np.asarray(key_points, dtype=np.float32)


def nn_dist(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.asarray([1e6], dtype=np.float32)
    out = []
    bs = 512
    b2 = np.sum(b * b, axis=1)[None, :]
    for i in range(0, len(a), bs):
        x = a[i : i + bs]
        d2 = np.sum(x * x, axis=1)[:, None] + b2 - 2.0 * x @ b.T
        out.append(np.sqrt(np.maximum(d2.min(axis=1), 0.0)))
    return np.concatenate(out)


def score_pair(gt, pr, template):
    gt_line, gt_pts = render_geom(gt, template)
    pr_line, pr_pts = render_geom(pr, template)
    d_pr_gt = nn_dist(pr_line, gt_line)
    d_gt_pr = nn_dist(gt_line, pr_line)
    point_d = nn_dist(gt_pts, pr_pts)
    reproj = float(0.5 * (d_pr_gt.mean() + d_gt_pr.mean()))
    out = {
        "point_acc": float((point_d <= 5).mean()) if len(point_d) else 0.0,
        "line_acc": float((d_gt_pr <= 5).mean()) if len(d_gt_pr) else 0.0,
        "reproj_mean": reproj,
        "MRE": reproj,
    }
    for t in THRESHOLDS:
        close_gt = float((d_gt_pr <= t).sum())
        close_pr = float((d_pr_gt <= t).sum())
        inter = 0.5 * (close_gt + close_pr)
        union = len(d_gt_pr) + len(d_pr_gt) - inter
        out[f"JaC@{t}"] = float(inter / union) if union > 0 else 1.0
    return out


def mean_rows(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def score_file(path, template, out_detail):
    scores = []
    with path.open(newline="", encoding="utf-8") as f, out_detail.open("w", newline="", encoding="utf-8") as g:
        reader = csv.DictReader(f)
        fields = ["query_item", "top1_synth_item", "point_acc", "line_acc", "reproj_mean", "MRE"] + [f"JaC@{t}" for t in THRESHOLDS]
        writer = csv.DictWriter(g, fieldnames=fields)
        writer.writeheader()
        for r in reader:
            gt = {
                "pan": float(r["gt_pan"]),
                "roll": float(r["gt_roll"]),
                "tilt": float(r["gt_tilt"]),
                "focal": float(r["gt_focal"]),
            }
            pr = {
                "pan": float(r["pred_pan"]),
                "roll": float(r["pred_roll"]),
                "tilt": float(r["pred_tilt"]),
                "focal": float(r["pred_focal"]),
            }
            s = score_pair(gt, pr, template)
            scores.append(s)
            writer.writerow({"query_item": r["query_item"], "top1_synth_item": r["top1_synth_item"], **s})
    return mean_rows(scores)


def main():
    template = load_template()
    for probe in PROBES:
        rows = []
        probe_dir = PREV6 / probe
        out_dir = probe_dir / "projection_metrics_20260713"
        out_dir.mkdir(parents=True, exist_ok=True)
        for method, train_n in RUNS:
            src = probe_dir / f"{method}_{train_n}_details.csv"
            if not src.exists():
                continue
            m = score_file(src, template, out_dir / f"{method}_{train_n}_projection_details.csv")
            rec = {"方法": method, "train_n": train_n, **m}
            rows.append(rec)
            (out_dir / f"{method}_{train_n}_projection_metrics.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            print(json.dumps({"probe": probe, **rec}, ensure_ascii=False), flush=True)
        keys = ["方法", "train_n", "point_acc", "line_acc", "reproj_mean", "MRE"] + [f"JaC@{t}" for t in THRESHOLDS]
        with (out_dir / "summary_projection_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out_dir / 'summary_projection_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
