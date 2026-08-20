#!/usr/bin/env python
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/plugins/calibration/sn_calibration_baseline")
from camera import Camera, rotation_matrix_to_pan_tilt_roll


REPO = "/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate"
NBJW = "/remote-home/jiayuanrao/yishan/datasets/SoccerNetGS_2024_nbjw/test"
REPORT = os.path.join(
    REPO,
    "outputs/gsr/temporal_hrnet/temporal_calib_results_hub/"
    "full_finetune_temporal_nbjw_k3_official_aux_20260701/"
    "report_eval_visual_20260701",
)
RUN_DIRS = [
    "test_stride5_scatter_reproj25_fullft_offaux_last_motion_k3",
    "test_stride5_scatter_reproj25_fullft_offaux_last_nomotion_k3",
    "test_stride5_scatter_reproj25_fullft_offaux_stage1_motion_k3",
    "test_stride5_scatter_reproj25_fullft_offaux_stage1_nomotion_k3",
]

PITCH_LINES = {
    # Official NBJW world coordinates from utils_linesWC.py.
    "Side line top": [(0.0, 0.0), (105.0, 0.0)],
    "Side line bottom": [(0.0, 68.0), (105.0, 68.0)],
    "Side line left": [(0.0, 0.0), (0.0, 68.0)],
    "Side line right": [(105.0, 0.0), (105.0, 68.0)],
    "Middle line": [(52.5, 0.0), (52.5, 68.0)],
    "Big rect. left top": [(0.0, 13.84), (16.5, 13.84)],
    "Big rect. left bottom": [(0.0, 54.16), (16.5, 54.16)],
    "Big rect. left main": [(16.5, 13.84), (16.5, 54.16)],
    "Big rect. right top": [(88.5, 13.84), (105.0, 13.84)],
    "Big rect. right bottom": [(88.5, 54.16), (105.0, 54.16)],
    "Big rect. right main": [(88.5, 13.84), (88.5, 54.16)],
    "Small rect. left top": [(0.0, 24.84), (5.5, 24.84)],
    "Small rect. left bottom": [(0.0, 43.16), (5.5, 43.16)],
    "Small rect. left main": [(5.5, 24.84), (5.5, 43.16)],
    "Small rect. right top": [(99.5, 24.84), (105.0, 24.84)],
    "Small rect. right bottom": [(99.5, 43.16), (105.0, 43.16)],
    "Small rect. right main": [(99.5, 24.84), (99.5, 43.16)],
}


def fit_line(points):
    arr = np.asarray(points, dtype=np.float64)
    if arr.shape[0] < 2:
        return None
    A = np.c_[arr[:, 0], arr[:, 1], np.ones(arr.shape[0])]
    _, _, vt = np.linalg.svd(A)
    line = vt[-1]
    n = np.linalg.norm(line[:2])
    return line / n if n > 0 else None


def intersect(l1, l2):
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float64)


def pitch_line(name):
    p1, p2 = PITCH_LINES[name]
    return fit_line([p1, p2])


def homography_dlt(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] < 4:
        return None
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(A))
    H = vt[-1].reshape(3, 3)
    if abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


def project(H, pts):
    pts = np.asarray(pts, dtype=np.float64)
    hp = np.c_[pts, np.ones(len(pts))] @ H.T
    ok = np.abs(hp[:, 2]) > 1e-9
    out = np.full((len(pts), 2), np.nan)
    out[ok] = hp[ok, :2] / hp[ok, 2:3]
    return out


def ransac_homography(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 4:
        return None, 0, float("inf")
    rng = random.Random(20260701)
    best = (None, np.zeros(len(src), dtype=bool), float("inf"))
    samples = min(500, max(80, len(src) * 20))
    idxs = list(range(len(src)))
    for _ in range(samples):
        sample = rng.sample(idxs, 4)
        H = homography_dlt(src[sample], dst[sample])
        if H is None:
            continue
        pred = project(H, src)
        err = np.linalg.norm(pred - dst, axis=1)
        inl = np.isfinite(err) & (err < 3.0)
        score = (int(inl.sum()), -float(np.nanmedian(err[inl])) if inl.any() else -1e9)
        best_score = (int(best[1].sum()), -best[2])
        if score > best_score:
            best = (H, inl, float(np.nanmedian(err[inl])) if inl.any() else float("inf"))
    H, inl, med = best
    if H is None or int(inl.sum()) < 4:
        H = homography_dlt(src, dst)
        if H is None:
            return None, 0, float("inf")
        err = np.linalg.norm(project(H, src) - dst, axis=1)
        return H, len(src), float(np.nanmedian(err))
    H2 = homography_dlt(src[inl], dst[inl])
    return (H2 if H2 is not None else H), int(inl.sum()), med


def frame_json(video, frame):
    return os.path.join(NBJW, f"3{int(video):03d}{int(frame):06d}.json")


def physical_angle(video, frame):
    path = frame_json(video, frame)
    if not os.path.exists(path):
        return None, "missing_json", 0, 0, None
    data = json.load(open(path, "r"))
    img_lines = {}
    world_lines = {}
    for name, pts in data.items():
        if name not in PITCH_LINES or not isinstance(pts, list) or len(pts) < 2:
            continue
        xy = [(float(p["x"]), float(p["y"])) for p in pts if "x" in p and "y" in p]
        if len(xy) < 2:
            continue
        il = fit_line(xy)
        wl = pitch_line(name)
        if il is not None and wl is not None:
            img_lines[name] = il
            world_lines[name] = wl

    src, dst = [], []
    names = sorted(img_lines)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pi = intersect(img_lines[a], img_lines[b])
            pw = intersect(world_lines[a], world_lines[b])
            if pi is None or pw is None:
                continue
            if not (-0.5 <= pi[0] <= 1.5 and -0.5 <= pi[1] <= 1.5):
                continue
            src.append(pi)
            dst.append(pw)
    if len(src) < 4:
        return None, "few_intersections", len(names), len(src), None
    H, ninl, mederr = ransac_homography(np.asarray(src), np.asarray(dst))
    if H is None:
        return None, "homography_failed", len(names), len(src), None

    # H maps normalized image -> pitch. Convert it to pitch -> pixel image and
    # use the project's camera decomposition, which is closer to camera optical
    # axis than joining two arbitrary image points on the ground plane.
    norm_to_pix = np.array([[1920.0, 0.0, 0.0], [0.0, 1080.0, 0.0], [0.0, 0.0, 1.0]])
    H_pix_to_pitch = H @ np.linalg.inv(norm_to_pix)
    try:
        H_pitch_to_pix = np.linalg.inv(H_pix_to_pitch)
    except np.linalg.LinAlgError:
        return None, "homography_singular", len(names), len(src), None
    cam = Camera(1920, 1080)
    if not cam.from_homography(H_pitch_to_pix):
        return None, "camera_decompose_failed", len(names), len(src), None
    pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(cam.rotation)
    pan_deg = math.degrees(pan)
    # In sn_calibration_baseline camera convention, pan=0 is parallel to the
    # pitch y-axis, i.e. the Middle line direction. Fold 180-degree ambiguity.
    ang = pan_deg
    while ang > 90:
        ang -= 180
    while ang < -90:
        ang += 180
    return ang, "ok", len(names), len(src), {
        "inliers": ninl,
        "median_pitch_err": mederr,
        "camera_pan_deg": pan_deg,
        "camera_tilt_deg": math.degrees(tilt),
        "camera_roll_deg": math.degrees(roll),
    }


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def plot_one(out_dir, run_name, rows):
    rows = [r for r in rows if r.get("angle_status") == "ok" and r.get("reproj_mean")]
    rows = [r for r in rows if float(r["reproj_mean"]) < 25]
    if not rows:
        return
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)

    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=160)
    colors = {"baseline": "#222222", run_name: "#d62728"}
    for name in ["baseline", run_name]:
        pts = by_run.get(name, [])
        if not pts:
            continue
        x = [float(r["physical_signed_angle_deg"]) for r in pts]
        y = [float(r["reproj_mean"]) for r in pts]
        ax.scatter(x, y, s=13, alpha=0.55 if name == "baseline" else 0.75,
                   c=colors.get(name, "#1f77b4"), label=f"{name} n={len(pts)}",
                   edgecolors="none")
    ax.axvline(0, color="#555555", lw=1.0, ls="--")
    ax.set_xlim(-90, 90)
    ax.set_ylim(0, 25)
    ax.set_xlabel("Physical signed angle to Middle line (deg)")
    ax.set_ylabel("reproj_mean (<25)")
    ax.set_title(f"Reprojection vs physical angle: {run_name}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scatter_baseline_vs_model_physical_angle.png"))
    plt.close(fig)

    for name in ["baseline", run_name]:
        pts = by_run.get(name, [])
        if not pts:
            continue
        x = np.array([float(r["physical_signed_angle_deg"]) for r in pts])
        y = np.array([float(r["reproj_mean"]) for r in pts])
        fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=160)
        h = ax.hist2d(x, y, bins=[36, 25], range=[[-90, 90], [0, 25]], cmap="magma")
        ax.axvline(0, color="white", lw=1.0, ls="--")
        ax.set_xlabel("Physical signed angle to Middle line (deg)")
        ax.set_ylabel("reproj_mean (<25)")
        ax.set_title(f"Frequency heatmap: {name}")
        fig.colorbar(h[3], ax=ax, label="frame count")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"frequency_heatmap_{name}_physical_angle.png"))
        plt.close(fig)


def main():
    all_diag = []
    angle_cache = {}
    for run_dir in RUN_DIRS:
        root = os.path.join(REPORT, run_dir)
        src_csv = os.path.join(root, "test_frame_scores.csv")
        if not os.path.exists(src_csv):
            print("missing", src_csv)
            continue
        rows = read_csv(src_csv)
        run_names = sorted({r["run"] for r in rows if r["run"] != "baseline"})
        run_name = run_names[0] if run_names else run_dir.replace("test_stride5_scatter_reproj25_", "")
        out_dir = os.path.join(root, "angle_reproj_scatter_physical")
        os.makedirs(out_dir, exist_ok=True)
        out_rows = []
        status_counts = defaultdict(int)
        for r in rows:
            key = (r["video"], r["frame"])
            if key not in angle_cache:
                angle_cache[key] = physical_angle(r["video"], r["frame"])
            angle, status, nlines, nints, extra = angle_cache[key]
            status_counts[status] += 1
            rr = dict(r)
            rr["physical_signed_angle_deg"] = "" if angle is None else f"{angle:.6f}"
            rr["angle_status"] = status
            rr["angle_fit_lines"] = str(nlines)
            rr["angle_intersections"] = str(nints)
            rr["angle_inliers"] = "" if not extra else str(extra["inliers"])
            rr["angle_median_pitch_err"] = "" if not extra else f"{extra['median_pitch_err']:.6f}"
            rr["gt_camera_pan_deg"] = "" if not extra else f"{extra['camera_pan_deg']:.6f}"
            rr["gt_camera_tilt_deg"] = "" if not extra else f"{extra['camera_tilt_deg']:.6f}"
            rr["gt_camera_roll_deg"] = "" if not extra else f"{extra['camera_roll_deg']:.6f}"
            out_rows.append(rr)
        fields = list(out_rows[0].keys())
        write_csv(os.path.join(out_dir, "test_frame_scores_with_physical_angle.csv"), out_rows, fields)
        plot_one(out_dir, run_name, out_rows)

        ok_angles = [float(r["physical_signed_angle_deg"]) for r in out_rows
                     if r["angle_status"] == "ok" and r["physical_signed_angle_deg"]]
        diag = {
            "run_dir": run_dir,
            "run_name": run_name,
            "rows": len(out_rows),
            "status_counts": dict(status_counts),
            "angle_quantiles": {},
        }
        if ok_angles:
            arr = np.asarray(ok_angles)
            diag["angle_quantiles"] = {
                "p05": float(np.percentile(arr, 5)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
            }
        with open(os.path.join(out_dir, "physical_angle_diagnostics.json"), "w") as f:
            json.dump(diag, f, indent=2)
        with open(os.path.join(out_dir, "README.md"), "w") as f:
            f.write("# Physical Angle Replot\n\n")
            f.write("- Reuses `test_frame_scores.csv`; no model inference or calibration eval rerun.\n")
            f.write("- `reproj_mean` is copied from the existing CSV.\n")
            f.write("- Angle is recomputed from GT field-line JSON by fitting image-to-pitch homography from line-line intersections, then decomposing it with `sn_calibration_baseline.Camera.from_homography()`.\n")
            f.write("- 0 deg means the decomposed camera pan is parallel to the SoccerPitch Middle line axis in the project camera convention; sign separates left/right yaw.\n")
            f.write("- Scatter filters `reproj_mean < 25`.\n")
        all_diag.append(diag)
        print(run_dir, json.dumps(diag, sort_keys=True))

    out_summary = os.path.join(REPORT, "physical_angle_replot_summary_20260701.json")
    with open(out_summary, "w") as f:
        json.dump(all_diag, f, indent=2)
    print("summary", out_summary)


if __name__ == "__main__":
    main()
