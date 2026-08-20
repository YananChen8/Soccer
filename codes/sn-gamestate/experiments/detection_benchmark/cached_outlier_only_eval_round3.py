"""Outlier-only temporal adapter evaluation for round 3.

Baseline keypoints are kept unchanged except RANSAC-rejected keys. For those
keys, coordinates from a temporal adapter are substituted, RANSAC is run again,
and keys still rejected are deleted before final camera solving.
"""
import argparse
import copy
import glob
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from cached_full_test_round2 import (
    DATA_ROOT,
    KeypointKalman,
    _score_job,
    load_adapter,
    ransac_inlier_keys,
    solve_params,
    smoothness,
)
from nbjw_calib.utils.utils_heatmap import (
    complete_keypoints,
    coords_to_dict,
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
)
from sn_gamestate.temporal_hrnet import pad_window


def decode_keypoints(kp_hm, line_hm):
    kc = get_keypoints_from_heatmap_batch_maxpool(kp_hm[:, :-1])
    lc = get_keypoints_from_heatmap_batch_maxpool_l(line_hm[:, :-1])
    return complete_keypoints(
        coords_to_dict(kc, threshold=0.1449),
        coords_to_dict(lc, threshold=0.2983),
        w=960,
        h=540,
        normalize=True,
    )[0]


def merge_outlier_only(base, refined, mode, kp_filter):
    before = set(base)
    inliers = ransac_inlier_keys(base)
    outliers = before - inliers
    merged = copy.deepcopy(base)
    replaced = 0
    for key in outliers:
        if mode == "adapter":
            if key in refined:
                merged[key]["x"] = refined[key]["x"]
                merged[key]["y"] = refined[key]["y"]
                replaced += 1
        elif mode == "kalman":
            pred = kp_filter.predict(key)
            if pred is not None:
                merged[key]["x"], merged[key]["y"] = pred
                replaced += 1
        else:
            raise ValueError(mode)
    still = set(merged) - ransac_inlier_keys(merged)
    for key in still:
        merged.pop(key, None)
    return merged, {
        "initial_outliers": len(outliers),
        "replaced": replaced,
        "removed": len(still),
        "detected_points": len(before),
    }


def solve_video(args):
    vid, name, ckpt, cache_root, stride, mode, device = args
    fwd, k = load_adapter(ckpt, device) if ckpt else (None, 1)
    files = sorted(glob.glob(f"{cache_root}/test/SNGS-{vid}/frame_*.npz"))
    buf, out = [], {}
    kp_filter = KeypointKalman()
    stats = defaultdict(int)
    for i, f in enumerate(files):
        try:
            d = np.load(f)
            frame = int(d["frame"])
            kp = torch.from_numpy(d["kp_hm"].astype(np.float32)).unsqueeze(0).to(device)
            line = torch.from_numpy(d["line_hm"].astype(np.float32)).unsqueeze(0).to(device)
        except Exception:
            stats["bad_cache"] += 1
            continue
        if fwd is not None:
            buf.append(kp)
            if len(buf) > k:
                buf.pop(0)
        if i % stride != 0:
            continue
        with torch.no_grad():
            base = decode_keypoints(kp, line)
            if mode == "baseline":
                final = base
            elif mode == "kalman":
                final, st = merge_outlier_only(base, base, "kalman", kp_filter)
                for key, val in st.items():
                    stats[key] += val
            elif mode == "adapter":
                refined_hm = fwd(pad_window(torch.stack(buf, 1), k))
                refined = decode_keypoints(refined_hm, line)
                final, st = merge_outlier_only(base, refined, "adapter", kp_filter)
                for key, val in st.items():
                    stats[key] += val
            else:
                raise ValueError(mode)
        params = solve_params(final)
        kp_filter.update_many(final)
        out[f"3{vid}{frame % 1000000:06d}"] = params
    return name, vid, out, smoothness(out), dict(stats)


def parse_adapter_specs(specs):
    items = {}
    for spec in specs or []:
        name, path = spec.split("=", 1)
        items[name] = path
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--adapter", action="append", default=[], help="NAME=PATH")
    ap.add_argument("--videos", nargs="+", default=[str(x) for x in list(range(116, 151)) + list(range(187, 201))])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapters = parse_adapter_specs(args.adapter)
    jobs = []
    names = {"baseline": (None, "baseline"), "kalman_keypoint": (None, "kalman")}
    names.update({f"{name}_outlier_only": (path, "adapter") for name, path in adapters.items()})
    for name, (path, mode) in names.items():
        for vid in args.videos:
            jobs.append((vid, name, path, args.cache_root, args.stride, mode, args.device))

    params = {name: {} for name in names}
    sm = {name: [] for name in names}
    aux = {name: defaultdict(int) for name in names}
    with Pool(min(args.nproc, len(jobs))) as pool:
        for name, vid, pb, s, st in pool.imap_unordered(solve_video, jobs):
            params[name][vid] = pb
            sm[name].append(s)
            for key, val in st.items():
                aux[name][key] += val
            print(f"params adapter={name} video={vid} frames={len(pb)}", flush=True)

    json.dump(params, open(out_dir / "params.json", "w"))
    score_jobs = [(vid, ad, params[ad][vid]) for ad in params for vid in params[ad]]
    with Pool(args.nproc) as pool:
        parts = pool.map(_score_job, score_jobs)
    agg = {a: {"micro": [], "macro": [], "reproj": [], "nvid": 0} for a in params}
    for ad, _vid, mi, ma, rj in parts:
        agg[ad]["micro"] += mi
        agg[ad]["macro"] += ma
        agg[ad]["reproj"] += rj
        agg[ad]["nvid"] += 1

    res = {}
    for ad, g in agg.items():
        mean_sm = [x["mean"] for x in sm[ad] if x["mean"] is not None]
        p95_sm = [x["p95"] for x in sm[ad] if x["p95"] is not None]
        res[ad] = {
            "point_acc": float(np.mean(g["micro"])) if g["micro"] else None,
            "line_acc": float(np.mean(g["macro"])) if g["macro"] else None,
            "reproj_mean": float(np.mean(g["reproj"])) if g["reproj"] else None,
            "n_frames": len(g["micro"]),
            "n_videos": g["nvid"],
            "smoothness_mean": float(np.mean(mean_sm)) if mean_sm else None,
            "smoothness_p95": float(np.mean(p95_sm)) if p95_sm else None,
            "mode_stats": dict(aux[ad]),
        }
    json.dump(res, open(out_dir / "result.json", "w"), indent=2)

    def fmt(val, nd=4):
        return "NA" if val is None else f"{val:.{nd}f}"

    lines = [
        "| adapter | point | line | reproj | smooth_mean | smooth_p95 | frames | init_out | repl | rm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ad, r in sorted(res.items()):
        st = r["mode_stats"]
        lines.append(
            f"| {ad} | {fmt(r['point_acc'])} | {fmt(r['line_acc'])} | {fmt(r['reproj_mean'], 2)} | "
            f"{fmt(r['smoothness_mean'], 2)} | {fmt(r['smoothness_p95'], 2)} | {r['n_frames']} | "
            f"{st.get('initial_outliers', 0)} | {st.get('replaced', 0)} | {st.get('removed', 0)} |"
        )
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((out_dir / "RESULTS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
