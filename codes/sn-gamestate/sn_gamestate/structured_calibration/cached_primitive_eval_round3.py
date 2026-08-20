#!/usr/bin/env python3
"""Primitive refine four-metric evaluator for Round3.

Unlike run_ablation.py, this only solves frames that will be scored. It reports
the same table columns as the temporal calibration summaries:
point, line, reproj, smooth_mean, smooth_p95, frames.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
import sys
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PROJECT = Path("/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "plugins/calibration/nbjw_calib"))
sys.path.insert(0, str(PROJECT / "plugins/calibration"))

from SoccerNet.Evaluation.utils_calibration import (  # noqa: E402
    evaluate_camera_prediction,
    get_polylines,
    mirror_labels,
)
from nbjw_calib.utils.utils_calib import (  # noqa: E402
    keypoint_aux_world_coords_2D as A,
    keypoint_world_coords_2D as W,
    pan_tilt_roll_to_orientation as PTR,
)
from sn_gamestate.structured_calibration import metrics as M  # noqa: E402
from sn_gamestate.structured_calibration import primitive_mapping as PM  # noqa: E402
from sn_gamestate.structured_calibration import primitive_weighting as PW  # noqa: E402
from sn_gamestate.structured_calibration import weighted_solver as WS  # noqa: E402

WIDTH, HEIGHT, THRESHOLD = 1920, 1080, 10

VARIANTS = {
    "refine_unw": [],
    "full": ["conf", "line", "circle", "box", "goal", "field"],
    "noconf": ["line", "circle", "box", "goal", "field"],
    "line": ["conf", "line"],
    "circle": ["conf", "circle"],
    "circle_strict": ["conf", "circle"],
    "box": ["conf", "box"],
    "box_strict": ["conf", "box"],
    "goal": ["conf", "goal"],
    "field": ["conf", "field"],
}


def flatten_params(params):
    if not params:
        return None
    vals = []
    for key in ["pan_degrees", "tilt_degrees", "roll_degrees", "x_focal_length", "y_focal_length"]:
        vals.append(float(params.get(key, 0.0)))
    vals.extend(float(x) for x in params.get("position_meters", [0, 0, 0]))
    return np.array(vals, dtype=np.float64)


def smoothness(params_by_gid):
    prev, vals = None, []
    for gid in sorted(params_by_gid):
        cur = flatten_params(params_by_gid[gid])
        if cur is None:
            continue
        if prev is not None:
            vals.append(float(np.linalg.norm(cur - prev)))
        prev = cur
    if not vals:
        return {"mean": None, "p95": None}
    return {"mean": float(np.mean(vals)), "p95": float(np.percentile(vals, 95))}


def sampled_gt_ids(data_root: Path, vid: str, stride: int):
    gt = M.load_gt_lines_for_video(data_root, vid)
    items = sorted(gt, key=lambda x: int(x))
    return set(items[::stride])


def score_job(args):
    data_root, vid, method, params_by_gid = args
    gt = M.load_gt_lines_for_video(Path(data_root), vid)
    micro, macro, reproj = [], [], []
    for gid, params in params_by_gid.items():
        if gid not in gt or not isinstance(params, dict) or not params:
            continue
        try:
            pred = get_polylines(params, WIDTH, HEIGHT, sampling_factor=0.9)
        except Exception:
            continue

        def one(glines):
            c, _, r = evaluate_camera_prediction(pred, glines, THRESHOLD)
            mi = c[0, 0] / c.sum() if c.sum() > 0 else 0.0
            per_line = [np.mean([e <= THRESHOLD for e in errs]) for errs in r.values() if errs]
            ma = float(np.mean(per_line)) if per_line else 0.0
            rj = [e for errs in r.values() for e in errs]
            return mi, ma, rj

        m1 = one(gt[gid])
        try:
            m2 = one(mirror_labels(gt[gid]))
        except Exception:
            m2 = None
        best = m1 if (m2 is None or m1[0] >= m2[0]) else m2
        micro.append(best[0])
        macro.append(best[1])
        reproj += best[2]
    return method, vid, micro, macro, reproj


def solve_video(job):
    vid, videos_root, data_root, stride, variants, ransac_gate, ransac_thresh = job
    world = PM.attach_world_coords(W, A)
    cfg = PW.WeightConfig()
    circle_ids = set(PM.KP_TO_LINE["Circle central"]) | set(PM.KP_TO_LINE["Circle left"]) | set(PM.KP_TO_LINE["Circle right"])
    box_ids = set()
    for prim in ["left_penalty_box", "right_penalty_box", "left_goal_area", "right_goal_area"]:
        box_ids |= set(PM.PRIM_TO_KPS[prim])

    keep_ids = sampled_gt_ids(Path(data_root), vid, stride)
    z = zipfile.ZipFile(str(Path(videos_root) / "sn-gamestate.pklz"))
    df = pickle.loads(z.read(f"{vid}_image.pkl"))
    out = {"raw": {}, **{name: {} for name in variants}}
    stats = defaultdict(int)

    for idx, row in df.iterrows():
        image_id = str(row.get("id"))
        if image_id not in keep_ids:
            continue
        frame = int(image_id[-6:]) if len(image_id) >= 6 else int(idx)
        gid = image_id
        raw_params = row.get("parameters") if isinstance(row.get("parameters"), dict) else {}
        kps = row.get("keypoints") if isinstance(row.get("keypoints"), dict) else {}
        out["raw"][gid] = raw_params
        if not kps or not raw_params:
            for name in variants:
                out[name][gid] = raw_params
            stats["empty"] += 1
            continue
        if ransac_gate:
            inliers = WS.ransac_inliers(kps, world, ransac_thresh)
            stats["ransac_removed"] += len(set(kps) - set(inliers))
            kps = {i: v for i, v in kps.items() if i in inliers}
            if len(kps) < 4:
                for name in variants:
                    out[name][gid] = raw_params
                stats["too_few_after_ransac"] += 1
                continue
        _weights, dbg = PW.compute_weights(kps, world, cfg)
        comps = dbg["components"]
        for name, terms in variants.items():
            if name == "circle_strict":
                wts = PW.compose_weights_masked(comps, terms, circle_ids, base_for_other=0.0)
            elif name == "box_strict":
                wts = PW.compose_weights_masked(comps, terms, box_ids, base_for_other=0.0)
            else:
                wts = PW.compose_weights(comps, terms)
            out[name][gid] = WS.weighted_refine(raw_params, kps, wts, world, PTR, use_weights=True)
        stats["frames"] += 1
        if stats["frames"] % 25 == 0:
            print(f"solve video={vid} frames={stats['frames']} last_frame={frame}", flush=True)

    sm = {name: smoothness(vals) for name, vals in out.items()}
    print(f"done video={vid} scored_frames={len(out['raw'])}", flush=True)
    return vid, out, sm, dict(stats)


def aggregate_results(params, smooth, score_parts):
    agg = {m: {"micro": [], "macro": [], "reproj": [], "nvid": 0} for m in params}
    for method, _vid, mi, ma, rj in score_parts:
        agg[method]["micro"] += mi
        agg[method]["macro"] += ma
        agg[method]["reproj"] += rj
        agg[method]["nvid"] += 1
    results = {}
    for method, g in agg.items():
        mean_sm = [x["mean"] for x in smooth[method] if x["mean"] is not None]
        p95_sm = [x["p95"] for x in smooth[method] if x["p95"] is not None]
        results[method] = {
            "point_acc": float(np.mean(g["micro"])) if g["micro"] else None,
            "line_acc": float(np.mean(g["macro"])) if g["macro"] else None,
            "reproj_mean": float(np.mean(g["reproj"])) if g["reproj"] else None,
            "smoothness_mean": float(np.mean(mean_sm)) if mean_sm else None,
            "smoothness_p95": float(np.mean(p95_sm)) if p95_sm else None,
            "n_frames": len(g["micro"]),
            "n_videos": g["nvid"],
        }
    return results


def fmt(val, nd=4):
    return "NA" if val is None else f"{val:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-root", default=str(PROJECT / "outputs/gsr/calib_baseline_test/nbjw/states"))
    ap.add_argument("--data-root", default=str(PROJECT / "datasets/SoccerNetGS/valid"))
    ap.add_argument("--out-dir", default=str(PROJECT / "outputs/gsr/structured_calib/results/round3_primitive_four_metrics"))
    ap.add_argument("--videos", nargs="+", default=["021", "023", "034", "040", "041", "051", "052", "085", "091", "093"])
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--variants", nargs="+", default=["refine_unw", "full", "box", "goal", "circle", "box_strict", "circle_strict", "field"])
    ap.add_argument("--ransac-gate", action="store_true")
    ap.add_argument("--ransac-thresh", type=float, default=30.0)
    args = ap.parse_args()

    variants = {name: VARIANTS[name] for name in args.variants}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(v, args.states_root, args.data_root, args.stride, variants, args.ransac_gate, args.ransac_thresh) for v in args.videos]

    params = {"raw": {}}
    params.update({name: {} for name in variants})
    smooth = {name: [] for name in params}
    stats = {}
    with Pool(min(args.nproc, len(jobs))) as pool:
        for vid, out, sm, st in pool.imap_unordered(solve_video, jobs):
            stats[vid] = st
            for method, vals in out.items():
                params[method][vid] = vals
                smooth[method].append(sm[method])

    json.dump(params, open(out_dir / "params.json", "w"))
    json.dump({"videos": args.videos, "stride": args.stride, "variants": list(params), "stats": stats}, open(out_dir / "run_meta.json", "w"), indent=2)

    score_jobs = [(args.data_root, vid, method, params[method][vid]) for method in params for vid in params[method]]
    with Pool(min(args.nproc, len(score_jobs))) as pool:
        score_parts = pool.map(score_job, score_jobs)

    results = aggregate_results(params, smooth, score_parts)
    json.dump(results, open(out_dir / "result.json", "w"), indent=2)

    order = ["raw"] + list(variants)
    lines = [
        "| method | point | line | reproj | smooth_mean | smooth_p95 | frames |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for method in order:
        r = results[method]
        lines.append(
            f"| {method} | {fmt(r['point_acc'])} | {fmt(r['line_acc'])} | {fmt(r['reproj_mean'], 2)} | "
            f"{fmt(r['smoothness_mean'], 2)} | {fmt(r['smoothness_p95'], 2)} | {r['n_frames']} |"
        )
        rows.append({
            "method": method,
            "point": r["point_acc"],
            "line": r["line_acc"],
            "reproj": r["reproj_mean"],
            "smooth_mean": r["smoothness_mean"],
            "smooth_p95": r["smoothness_p95"],
            "frames": r["n_frames"],
        })
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print((out_dir / "RESULTS.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
