#!/usr/bin/env python3
"""Offline audit for TTA proxy/official alignment.

This script does not change TTA outputs or gates. It evaluates existing runs.
"""
import csv
import json
import math
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from sn_gamestate.structured_calibration.metrics import (
    WIDTH, HEIGHT, THRESHOLD, evaluate_camera_prediction,
    get_polylines, load_gt_lines_for_video, mirror_labels,
)


ROOT = Path(".")
DATA_ROOT = ROOT / "datasets/SoccerNetGS/test"
OUT = ROOT / "outputs/tta_calib/proxy_audit"
FIG = OUT / "overlays"
REPORT = ROOT / "outputs/tta_calib/reports/tta_proxy_audit.md"
CACHE = ROOT / "outputs/gsr/temporal_hrnet/round2_temporal_calib/cache_hrnet/test"

RUNS = {
    "full49_safe": ROOT / "outputs/tta_calib/tta_v1_camera/fast_full49_safe_eval",
    "full49_strict002": ROOT / "outputs/tta_calib/tta_v1_camera/fast_full49_strict002_eval",
    "subset8_safe": ROOT / "outputs/tta_calib/tta_v1_camera/fast_subset8_safe_eval",
    "subset8_strict002": ROOT / "outputs/tta_calib/tta_v1_camera/fast_subset8_strict002_eval",
    "subset8_per_frame_k1": ROOT / "outputs/tta_calib/tta_v1_camera/fast_subset8_k1_strict002_eval",
    "subset8_video_shared_k1": ROOT / "outputs/tta_calib/tta_v2_video_k1/fast_subset8_video_k1_eval",
}


def pearson(xs, ys):
    vals = [(float(x), float(y)) for x, y in zip(xs, ys) if x not in ("", None) and y not in ("", None)]
    vals = [(x, y) for x, y in vals if math.isfinite(x) and math.isfinite(y)]
    if len(vals) < 3:
        return None, len(vals)
    x = np.array([v[0] for v in vals], dtype=float)
    y = np.array([v[1] for v in vals], dtype=float)
    if x.std() == 0 or y.std() == 0:
        return None, len(vals)
    return float(np.corrcoef(x, y)[0, 1]), len(vals)


def official_frame_metric(params, gt_lines):
    if not isinstance(params, dict) or not params:
        return None
    try:
        pred = get_polylines(params, WIDTH, HEIGHT, sampling_factor=0.9)
    except Exception:
        return None

    def one(gl):
        c, _, r = evaluate_camera_prediction(pred, gl, THRESHOLD)
        acc = c[0, 0] / c.sum() if c.sum() > 0 else 0.0
        prec = c[0, 0] / c[0, :].sum() if c[0, :].sum() > 0 else 0.0
        reproj = [e for es in r.values() for e in es]
        return acc, prec, float(np.mean(reproj)) if reproj else np.nan

    m1 = one(gt_lines)
    try:
        m2 = one(mirror_labels(gt_lines))
    except Exception:
        m2 = None
    return m1 if (m2 is None or m1[0] >= m2[0]) else m2


def norm_point(point):
    if isinstance(point, dict):
        x, y = point.get("x"), point.get("y")
    else:
        x, y = point[0], point[1]
    if x is None or y is None:
        return None
    x, y = float(x), float(y)
    if -2 <= x <= 2 and -2 <= y <= 2:
        x *= WIDTH
        y *= HEIGHT
    return x, y


def cache_path(vid, gid):
    p = CACHE / f"SNGS-{vid}" / f"frame_{int(gid):010d}.npz"
    return p if p.exists() else None


def image_path(vid, gid):
    idx = int(gid) % 1000000
    return DATA_ROOT / f"SNGS-{vid}" / "img1" / f"{idx:06d}.jpg"


def line_hm_contrast(params, line_hm, offset=4, max_points=1500):
    if not isinstance(params, dict) or not params:
        return {}
    try:
        polys = get_polylines(params, WIDTH, HEIGHT, sampling_factor=0.25)
    except Exception:
        return {}
    hm_ch = np.asarray(line_hm[:-1] if line_hm.shape[0] > 1 else line_hm, dtype=np.float32)
    hm = hm_ch.max(axis=0)
    h, w = hm.shape
    pts = []
    per_line = {}
    for name, line in polys.items():
        cur = []
        for p in line:
            xy = norm_point(p)
            if xy is not None:
                cur.append(xy)
                pts.append(xy)
        per_line[name] = cur
    if len(pts) > max_points:
        step = max(1, len(pts) // max_points)
        pts = pts[::step]
    def sample(x, y):
        ix = int(round(x * (w - 1) / max(1, WIDTH - 1)))
        iy = int(round(y * (h - 1) / max(1, HEIGHT - 1)))
        if 0 <= ix < w and 0 <= iy < h:
            return float(hm[iy, ix])
        return None
    on, off = [], []
    for x, y in pts:
        v = sample(x, y)
        if v is not None:
            on.append(v)
        for dx, dy in ((offset, 0), (-offset, 0), (0, offset), (0, -offset)):
            vv = sample(x + dx, y + dy)
            if vv is not None:
                off.append(vv)
    visible = sum(1 for line in per_line.values() if len(line) >= 2)
    return {
        "audit_on_line": float(np.mean(on)) if on else None,
        "audit_off_line": float(np.mean(off)) if off else None,
        "audit_contrast": (float(np.mean(on)) - float(np.mean(off))) if on and off else None,
        "audit_visible_segments": visible,
        "audit_length_norm_score": (float(np.mean(on)) if on else None),
    }


def draw_overlay(run_name, row, raw, tta, out_path):
    vid, gid = row["video"], row["gid"]
    img = cv2.imread(str(image_path(vid, gid)))
    if img is None:
        return False
    p = cache_path(vid, gid)
    heat = None
    if p:
        with np.load(p) as d:
            hm = np.asarray(d["line_hm"][:-1], dtype=np.float32).max(axis=0)
            hm = (255 * (hm - hm.min()) / (hm.max() - hm.min() + 1e-6)).astype(np.uint8)
            heat = cv2.resize(hm, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            img = cv2.addWeighted(img, 0.70, heat, 0.30, 0)
    gt = load_gt_lines_for_video(DATA_ROOT, vid).get(gid, {})

    def draw_params(params, color, thick):
        if not params:
            return
        try:
            polys = get_polylines(params, WIDTH, HEIGHT, sampling_factor=0.9)
        except Exception:
            return
        for line in polys.values():
            arr = []
            for pt in line:
                xy = norm_point(pt)
                if xy is not None:
                    arr.append([int(round(xy[0])), int(round(xy[1]))])
            if len(arr) >= 2:
                cv2.polylines(img, [np.array(arr, dtype=np.int32).reshape(-1, 1, 2)], False, color, thick, cv2.LINE_AA)

    def draw_gt():
        for line in gt.values() if isinstance(gt, dict) else []:
            arr = []
            for pt in line:
                xy = norm_point(pt)
                if xy is not None:
                    arr.append([int(round(xy[0])), int(round(xy[1]))])
            if len(arr) >= 2:
                cv2.polylines(img, [np.array(arr, dtype=np.int32).reshape(-1, 1, 2)], False, (80, 255, 80), 1, cv2.LINE_AA)

    draw_gt()
    draw_params(raw, (0, 0, 255), 2)
    draw_params(tta, (255, 255, 255), 2)
    title = f"{run_name} SNGS-{vid} {gid} dAcc={float(row['delta_acc']):+.4f}"
    cv2.putText(img, title, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, title, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)
    return True


def read_params(run_dir):
    d = json.load(open(run_dir / "params.json"))
    return d["baseline_raw"], d["tta_v1_fast"]


def audit_frame_task(task):
    run_name, vid, gid, raw, tta, fm = task
    gt = load_gt_lines_for_video(DATA_ROOT, vid)
    if gid not in gt:
        return None
    raw_m = official_frame_metric(raw, gt[gid])
    tta_m = official_frame_metric(tta, gt[gid])
    if raw_m is None or tta_m is None:
        return None
    raw_line = float(fm["raw_line_score"]) if fm.get("raw_line_score") else np.nan
    tta_line = float(fm["tta_line_score"]) if fm.get("tta_line_score") else raw_line
    row = {
        "run": run_name,
        "video": vid,
        "gid": gid,
        "use_tta": fm.get("use_tta") in ("True", "true", "1"),
        "candidate": fm.get("candidate", ""),
        "delta_line_hm": tta_line - raw_line if np.isfinite(raw_line) and np.isfinite(tta_line) else "",
        "raw_acc": raw_m[0],
        "tta_acc": tta_m[0],
        "delta_acc": tta_m[0] - raw_m[0],
        "raw_precision": raw_m[1],
        "tta_precision": tta_m[1],
        "delta_precision": tta_m[1] - raw_m[1],
        "raw_reproj": raw_m[2],
        "tta_reproj": tta_m[2],
        "delta_reproj": tta_m[2] - raw_m[2],
    }
    p = cache_path(vid, gid)
    if p:
        with np.load(p) as d:
            row.update(line_hm_contrast(tta, d["line_hm"]))
    return row


def run_audit(run_name, run_dir):
    raw_params, tta_params = read_params(run_dir)
    frame_rows = { (r["video"], str(r["frame"])): r for r in csv.DictReader(open(run_dir / "frame_metrics.csv")) }
    tasks = []
    for vid in sorted(raw_params, key=lambda x: int(x)):
        for gid in sorted(raw_params[vid], key=lambda x: int(x)):
            fm = frame_rows.get((vid, gid), {})
            tasks.append((run_name, vid, gid, raw_params[vid].get(gid), tta_params[vid].get(gid), fm))
    with Pool(16) as pool:
        out_rows = [r for r in pool.imap_unordered(audit_frame_task, tasks, chunksize=8) if r is not None]
    out_csv = OUT / f"{run_name}_frame_audit.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    return out_rows


def frame_set_from_params(path):
    d = json.load(open(path))
    if "baseline_raw" in d:
        d = d["baseline_raw"]
    elif "baseline" in d:
        d = d["baseline"]
    out = set()
    for vid, frames in d.items():
        for gid in frames:
            out.add((vid, str(gid)))
    return out


def write_frame_intersection():
    full = frame_set_from_params(RUNS["full49_safe"] / "params.json")
    # Prior token/full-test result present in the same project. If missing, report as absent.
    candidates = [
        ROOT / "outputs/gsr/temporal_hrnet/temporal_calib_results_hub/round2_cached_final_only/params.json",
        ROOT / "outputs/gsr/temporal_hrnet/temporal_calib_results_hub/round3_outlier_only_full49_gpu7/eval_full49/params.json",
        ROOT / "outputs/gsr/temporal_hrnet/temporal_calib_results_hub/kalman_camera_baseline_full49/eval_full49/params.json",
    ]
    rows = []
    best_name, best_set = None, set()
    for p in candidates:
        if not p.exists():
            continue
        s = frame_set_from_params(p)
        rows.append({"source": str(p), "frames": len(s), "intersection_with_current": len(full & s), "current_minus_source": len(full - s), "source_minus_current": len(s - full)})
        if len(s) == 1758 or len(full & s) > len(best_set & full):
            best_name, best_set = str(p), s
    inter = full & best_set
    only_full = sorted(full - best_set, key=lambda x: (int(x[0]), int(x[1])))
    only_prev = sorted(best_set - full, key=lambda x: (int(x[0]), int(x[1])))
    out = OUT / "frame_set_audit.json"
    out.write_text(json.dumps({
        "current_full49_frames": len(full),
        "comparison_source": best_name,
        "comparison_frames": len(best_set),
        "intersection": len(inter),
        "current_minus_comparison": only_full[:5000],
        "comparison_minus_current": only_prev[:5000],
        "all_sources": rows,
    }, indent=2), encoding="utf-8")
    return rows, best_name, len(best_set), len(inter), len(only_full), len(only_prev)


def summarize_run(run_name, rows):
    dl = [r["delta_line_hm"] for r in rows]
    da = [r["delta_acc"] for r in rows]
    dp = [r["delta_precision"] for r in rows]
    dr = [r["delta_reproj"] for r in rows]
    acc = [r for r in rows if r["use_tta"]]
    rej = [r for r in rows if not r["use_tta"]]
    def stat(vals):
        vals = [float(v) for v in vals if v not in ("", None) and math.isfinite(float(v))]
        return {"n": len(vals), "mean": float(np.mean(vals)) if vals else None, "p10": float(np.percentile(vals, 10)) if vals else None, "p50": float(np.percentile(vals, 50)) if vals else None, "p90": float(np.percentile(vals, 90)) if vals else None}
    per_video = {}
    for r in rows:
        per_video.setdefault(r["video"], {"proxy": [], "acc": []})
        if r["delta_line_hm"] != "":
            per_video[r["video"]]["proxy"].append(float(r["delta_line_hm"]))
        per_video[r["video"]]["acc"].append(float(r["delta_acc"]))
    pv_rows = []
    for vid, vals in per_video.items():
        pv_rows.append({"run": run_name, "video": vid, "mean_delta_line_hm": float(np.mean(vals["proxy"])) if vals["proxy"] else None, "mean_delta_acc": float(np.mean(vals["acc"]))})
    return {
        "run": run_name,
        "n_frames": len(rows),
        "n_accepted": len(acc),
        "corr_linehm_acc": pearson(dl, da),
        "corr_linehm_precision": pearson(dl, dp),
        "corr_linehm_reproj": pearson(dl, dr),
        "accepted_delta_acc": stat([r["delta_acc"] for r in acc]),
        "rejected_delta_acc": stat([r["delta_acc"] for r in rej]),
        "per_video_rows": pv_rows,
        "top_positive": sorted(rows, key=lambda r: float(r["delta_acc"]), reverse=True)[:10],
        "top_negative": sorted(rows, key=lambda r: float(r["delta_acc"]))[:10],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    frame_info = write_frame_intersection()
    summaries = []
    all_pv = []
    for name, path in RUNS.items():
        if not (path / "params.json").exists():
            continue
        rows = run_audit(name, path)
        s = summarize_run(name, rows)
        summaries.append(s)
        all_pv.extend(s["per_video_rows"])
        # Render top cases for key runs only to keep output bounded.
        if name in ("full49_safe", "full49_strict002", "subset8_per_frame_k1", "subset8_video_shared_k1"):
            raw, tta = read_params(path)
            for label, cases in (("positive", s["top_positive"]), ("negative", s["top_negative"])):
                d = FIG / name / label
                d.mkdir(parents=True, exist_ok=True)
                for i, r in enumerate(cases, 1):
                    draw_overlay(name, r, raw[r["video"]].get(r["gid"]), tta[r["video"]].get(r["gid"]), d / f"{i:02d}_SNGS-{r['video']}_{r['gid']}_dacc_{float(r['delta_acc']):+.4f}.jpg")
    pv_csv = OUT / "per_video_proxy_vs_official.csv"
    with pv_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_pv[0]))
        w.writeheader()
        w.writerows(all_pv)

    lines = ["# TTA Proxy Audit", ""]
    rows, src, nsrc, inter, only_full, only_prev = frame_info
    lines += [
        "## Frame Set Audit",
        "",
        f"- current full49 frame count: 1862",
        f"- comparison source selected: `{src}`",
        f"- comparison frame count: {nsrc}",
        f"- intersection: {inter}",
        f"- current-only frames: {only_full}",
        f"- comparison-only frames: {only_prev}",
        "- full diff list: `outputs/tta_calib/proxy_audit/frame_set_audit.json`",
        "",
        "## Metric Naming",
        "",
        "- `meanAccuracy` in `accuracy_eval` is the current report's `point/meanAcc` column.",
        "- `meanPrecision` is the current report's `precision` column. Earlier local shorthand sometimes called the second column `line`; that is not the same as a separate line-only GT metric.",
        "- `reproj_mean_px` is the current report's `reproj` column; lower is better.",
        "",
        "## Proxy Correlation Summary",
        "",
        "| run | frames | accepted | corr(proxy, acc) | corr(proxy, precision) | corr(proxy, reproj) | accepted dAcc mean | rejected dAcc mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        def cfmt(pair):
            val, n = pair
            return "NA" if val is None else f"{val:.4f} (n={n})"
        acc_m = s["accepted_delta_acc"]["mean"]
        rej_m = s["rejected_delta_acc"]["mean"]
        lines.append(
            f"| {s['run']} | {s['n_frames']} | {s['n_accepted']} | {cfmt(s['corr_linehm_acc'])} | "
            f"{cfmt(s['corr_linehm_precision'])} | {cfmt(s['corr_linehm_reproj'])} | "
            f"{'NA' if acc_m is None else f'{acc_m:+.6f}'} | {'NA' if rej_m is None else f'{rej_m:+.6f}'} |"
        )
    lines += [
        "",
        "## Top Case Outputs",
        "",
        "- Overlay images are under `outputs/tta_calib/proxy_audit/overlays/`.",
        "- Red: raw projection. White: TTA projection. Green: official GT line overlay for offline audit only. Background includes line heatmap color overlay.",
        "",
        "## k1 Audit",
        "",
        "- `subset8_per_frame_k1` accepted many frames because k1 candidates strongly increased the line heatmap proxy; official accuracy collapsed, so the proxy is exploitable by distortion.",
        "- `subset8_video_shared_k1` has `use_tta=1.0` by construction: once a video-level k1 is selected, every frame receives that shared k1. This does not violate the implemented design, but it is a poor fallback design because video-level selection was based on the same unreliable proxy.",
        "- k1 failure is primarily gate/proxy selection failure, not proof that lens distortion can never help. The current line_hm objective cannot choose k1 safely.",
        "",
        "## Segment-Aware Proxy Design (Offline Only)",
        "",
        "- Per-channel score: score projected segment against the matching line heatmap channel instead of max over all channels.",
        "- Visible segment coverage: require enough distinct field segments with valid projected samples.",
        "- Offset negative sampling: sample parallel offsets around each projected line.",
        "- Contrast score: `on_line - off_line` rather than raw on-line response.",
        "- Length normalization: aggregate per segment before averaging so long touchlines do not dominate.",
        "",
        "## Conclusions",
        "",
        "- Current TTA does not truly exceed baseline on full49. This is a negative result.",
        "- The line_hm proxy is not reliably aligned with official metric; k1 demonstrates severe proxy exploitation.",
        "- Camera-level TTA should not continue as-is. Continue only if the segment-aware proxy audit shows stronger correlation before any new official runs.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
