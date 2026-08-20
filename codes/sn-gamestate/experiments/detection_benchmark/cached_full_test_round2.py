"""Cached full-test calibration evaluation for temporal adapter round 2."""
import argparse
import copy
import glob
import json
import math
from collections import defaultdict, deque
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch

from nbjw_calib.utils.utils_calib import FramebyFrameCalib
from nbjw_calib.utils.utils_heatmap import (
    complete_keypoints,
    coords_to_dict,
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
)
from sn_gamestate.temporal_hrnet import (
    KeypointTokenTemporalAdapter,
    TemporalHeatmapAdapter,
    heatmaps_to_tokens,
    pad_window,
)


SNG = "/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate"
DATA_ROOT = f"{SNG}/datasets/SoccerNetGS/test"
ROUND2 = f"{SNG}/outputs/gsr/temporal_hrnet/round2_temporal_calib"
OLD = f"{SNG}/outputs/gsr/temporal_hrnet/quick_subset12"


class ScalarKalman:
    def __init__(self, q=1e-3, r=1.0):
        self.x = None
        self.p = 1.0
        self.q = q
        self.r = r

    def predict(self):
        if self.x is None:
            return None
        self.p += self.q
        return self.x

    def update(self, z):
        if self.x is None:
            self.x = float(z)
            return self.x
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (float(z) - self.x)
        self.p = (1.0 - k) * self.p
        return self.x


class KeypointKalman:
    def __init__(self):
        self.filters = defaultdict(lambda: (ScalarKalman(r=25.0), ScalarKalman(r=25.0)))

    def predict(self, key):
        fx, fy = self.filters[key]
        px, py = fx.predict(), fy.predict()
        return None if px is None or py is None else (px, py)

    def update_many(self, keypoints):
        for key, val in keypoints.items():
            fx, fy = self.filters[key]
            fx.update(val["x"])
            fy.update(val["y"])


def load_adapter(path, device):
    if path is None:
        return None, 1
    ck = torch.load(path, map_location=device)
    if ck.get("which") == "kp_token":
        a = KeypointTokenTemporalAdapter(
            channels=ck["channels"],
            window_size=ck["window_size"],
            architecture=ck["architecture"],
            hidden=ck.get("hidden", 64),
            residual_scale=ck.get("residual_scale", 1.0),
            max_shift_px=ck.get("max_shift_px", 12.0),
        )
        a.load_state_dict(ck["state_dict"])
        a.to(device).eval()

        def fwd(win):
            toks = torch.stack([heatmaps_to_tokens(win[:, t]) for t in range(win.shape[1])], dim=1)
            return a(toks, win[:, -1])[0]

        return fwd, ck["window_size"]
    a = TemporalHeatmapAdapter(
        ck["channels"],
        ck["window_size"],
        residual_scale=ck.get("residual_scale", 1.0),
        adapter_type=ck.get("adapter_type", "depthwise_conv3d"),
        mix_hidden=ck.get("mix_hidden", 128),
    )
    a.load_state_dict(ck["state_dict"])
    a.to(device).eval()
    return (lambda win: a(win)[0]), ck["window_size"]


def adapter_map():
    items = {
        "baseline": None,
        "old_stgcn_k50_ms12_ep1": f"{OLD}/stgcn_k50/kp_adapter_stgcn_k50.pt",
        "old_transformer_k50_ms12_ep1": f"{OLD}/transformer_k50/kp_adapter_transformer_k50.pt",
    }
    for arch in ["transformer", "stgcn"]:
        for epoch in [1, 2, 3]:
            p = f"{ROUND2}/checkpoints/{arch}_k50_ms5_rs1_ep3/kp_adapter_{arch}_k50_ms5_epoch{epoch}.pt"
            items[f"round2_{arch}_k50_ms5_ep{epoch}"] = p
    return {k: v for k, v in items.items() if v is None or Path(v).exists()}


def parse_adapter_specs(specs):
    items = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"adapter spec must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        if not name or not path:
            raise ValueError(f"adapter spec must be NAME=PATH, got {spec!r}")
        items[name] = path
    return items


def ransac_inlier_keys(keypoints):
    cam = FramebyFrameCalib(1920, 1080, denormalize=True)
    cam.update(copy.deepcopy(keypoints))
    cam.get_homography_from_ground_plane(use_ransac=50, inverse=True)
    keys = set()
    for group in getattr(cam, "key_pts", []) or []:
        keys.update(group)
    return keys


def solve_params(keypoints, use_ransac=50):
    cam = FramebyFrameCalib(1920, 1080, denormalize=True)
    cam.update(copy.deepcopy(keypoints))
    h = cam.get_homography_from_ground_plane(use_ransac=use_ransac, inverse=True)
    if h is None:
        return {}
    try:
        vr = cam.heuristic_voting()
        return vr["cam_params"] if vr else {}
    except Exception:
        return {}


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


def camera_kalman_filter(params, history, filters):
    cur = flatten_params(params)
    if cur is None:
        return params, False
    pred_vals = []
    for i, f in enumerate(filters):
        p = f.predict()
        pred_vals.append(cur[i] if p is None else p)
    pred = np.array(pred_vals)
    dist = float(np.linalg.norm(cur - pred))
    scale = np.median(np.abs(np.array(history) - np.median(history))) * 3.0 if len(history) >= 8 else math.inf
    changed = dist > max(scale, 25.0)
    final = pred if changed else cur
    for i, f in enumerate(filters):
        f.update(final[i])
    history.append(float(np.linalg.norm(final - pred)))
    out = dict(params)
    for key, val in zip(["pan_degrees", "tilt_degrees", "roll_degrees", "x_focal_length", "y_focal_length"], final[:5]):
        out[key] = float(val)
    out["position_meters"] = [float(x) for x in final[5:8]]
    return out, changed


def solve_video(args):
    vid, name, ckpt, cache_root, stride, mode, device = args
    fwd, k = load_adapter(ckpt, device)
    files = sorted(glob.glob(f"{cache_root}/test/SNGS-{vid}/frame_*.npz"))
    buf, out = [], {}
    kp_filter = KeypointKalman()
    cam_filters = [ScalarKalman() for _ in range(8)]
    cam_history = deque(maxlen=24)
    stats = {"keypoint_replaced": 0, "keypoint_removed": 0, "camera_replaced": 0, "bad_cache": 0}
    bad_files = []
    for i, f in enumerate(files):
        if fwd is None and i % stride != 0:
            continue
        try:
            d = np.load(f)
            kp_arr = d["kp_hm"].astype(np.float32)
            line_arr = d["line_hm"].astype(np.float32)
            frame = int(d["frame"])
        except Exception as exc:
            stats["bad_cache"] += 1
            bad_files.append(f"{f}\t{type(exc).__name__}: {exc}")
            continue
        kp = torch.from_numpy(kp_arr).unsqueeze(0).to(device)
        if fwd is not None:
            buf.append(kp)
            if len(buf) > k:
                buf.pop(0)
        if i % stride != 0:
            continue
        line = torch.from_numpy(line_arr).unsqueeze(0).to(device)
        with torch.no_grad():
            refined = fwd(pad_window(torch.stack(buf, 1), k)) if fwd is not None else kp
        kc = get_keypoints_from_heatmap_batch_maxpool(refined[:, :-1])
        lc = get_keypoints_from_heatmap_batch_maxpool_l(line[:, :-1])
        pred = complete_keypoints(
            coords_to_dict(kc, threshold=0.1449),
            coords_to_dict(lc, threshold=0.2983),
            w=960,
            h=540,
            normalize=True,
        )[0]
        if mode == "keypoint":
            before = set(pred)
            inliers = ransac_inlier_keys(pred)
            outliers = before - inliers
            for key in list(outliers):
                p = kp_filter.predict(key)
                if p is not None:
                    pred[key]["x"], pred[key]["y"] = p
                    stats["keypoint_replaced"] += 1
            still = before - ransac_inlier_keys(pred)
            for key in still:
                pred.pop(key, None)
                stats["keypoint_removed"] += 1
        params = solve_params(pred)
        if mode == "camera" and params:
            params, changed = camera_kalman_filter(params, cam_history, cam_filters)
            stats["camera_replaced"] += int(changed)
        kp_filter.update_many(pred)
        out[f"3{vid}{frame % 1000000:06d}"] = params
    return name, vid, out, smoothness(out), stats, bad_files


def _score_job(args):
    vid, adapter, pb = args
    from sn_gamestate.structured_calibration.metrics import (
        HEIGHT,
        THRESHOLD,
        WIDTH,
        evaluate_camera_prediction,
        get_polylines,
        load_gt_lines_for_video,
        mirror_labels,
    )
    gt = load_gt_lines_for_video(DATA_ROOT, vid)
    micro, macro, reproj = [], [], []
    for gid, params in pb.items():
        if gid not in gt or not isinstance(params, dict) or not params:
            continue
        try:
            pred = get_polylines(params, WIDTH, HEIGHT, sampling_factor=0.9)
        except Exception:
            continue

        def both(gl):
            c, _, r = evaluate_camera_prediction(pred, gl, THRESHOLD)
            mi = c[0, 0] / c.sum() if c.sum() > 0 else 0.0
            pl = [np.mean([e <= THRESHOLD for e in es]) for es in r.values() if es]
            ma = float(np.mean(pl)) if pl else 0.0
            rj = [e for es in r.values() for e in es]
            return mi, ma, rj

        m1 = both(gt[gid])
        try:
            m2 = both(mirror_labels(gt[gid]))
        except Exception:
            m2 = None
        m = m1 if (m2 is None or m1[0] >= m2[0]) else m2
        micro.append(m[0])
        macro.append(m[1])
        reproj += m[2]
    return adapter, vid, micro, macro, reproj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--nproc", type=int, default=48)
    ap.add_argument("--mode", choices=["none", "keypoint", "camera"], default="none")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--adapter", action="append", default=[], help="Extra adapter as NAME=PATH")
    ap.add_argument("--videos", nargs="+", default=[str(x) for x in list(range(116, 151)) + list(range(187, 201))])
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = adapter_map()
    names.update(parse_adapter_specs(args.adapter))
    if args.only:
        wanted = set(args.only)
        names = {k: v for k, v in names.items() if k in wanted}
        missing = sorted(wanted - set(names))
        if missing:
            raise SystemExit(f"unknown adapters: {missing}; available={sorted(adapter_map())}")
    jobs = [(v, n, p, args.cache_root, args.stride, args.mode, args.device) for n, p in names.items() for v in args.videos]
    params = {n: {} for n in names}
    sm = {n: [] for n in names}
    aux = {n: defaultdict(int) for n in names}
    bad_cache = []
    with Pool(min(args.nproc, len(jobs))) as pool:
        for name, vid, pb, s, st, bad_files in pool.imap_unordered(solve_video, jobs):
            params[name][vid] = pb
            sm[name].append(s)
            for k, v in st.items():
                aux[name][k] += v
            bad_cache.extend(f"{name}\t{vid}\t{x}" for x in bad_files)
            print(f"params mode={args.mode} adapter={name} video={vid} frames={len(pb)}", flush=True)
    if bad_cache:
        (out_dir / "bad_cache_files.txt").write_text("\n".join(bad_cache) + "\n", encoding="utf-8")
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
    lines = ["| adapter | point | line | reproj | smooth_mean | smooth_p95 | frames |", "|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(val, nd=4):
        return "NA" if val is None else f"{val:.{nd}f}"
    for ad, r in sorted(res.items()):
        lines.append(
            f"| {ad} | {fmt(r['point_acc'])} | {fmt(r['line_acc'])} | "
            f"{fmt(r['reproj_mean'], 2)} | {fmt(r['smoothness_mean'], 2)} | {fmt(r['smoothness_p95'], 2)} | {r['n_frames']} |"
        )
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((out_dir / "RESULTS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
