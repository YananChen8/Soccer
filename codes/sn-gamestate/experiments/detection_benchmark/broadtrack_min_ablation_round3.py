"""Minimal BroadTrack-inspired calibration ablation.

This is not a BroadTrack reimplementation. It isolates three mechanisms on top
of cached NBJW HRNet heatmaps:

- radial_k1: solve cameras with cv2.calibrateCamera allowing K1 radial distortion.
- tripod: gate camera-center jumps toward a rolling fixed-tripod estimate.
- flow: use LK optical flow to replace only RANSAC-rejected keypoints, then rerun RANSAC.
"""
import argparse
import copy
import glob
import json
from collections import defaultdict, deque
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch

from cached_full_test_round2 import DATA_ROOT, _score_job, flatten_params, ransac_inlier_keys, smoothness
from nbjw_calib.utils.utils_calib import FramebyFrameCalib, rotation_matrix_to_pan_tilt_roll
from nbjw_calib.utils.utils_heatmap import (
    complete_keypoints,
    coords_to_dict,
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
)


class RadialK1Calib(FramebyFrameCalib):
    def get_cam_params(self, mode="full", use_ransac=0, refine=False):
        flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_FIX_TANGENT_DIST | cv2.CALIB_FIX_S1_S2_S3_S4 | cv2.CALIB_FIX_TAUX_TAUY
        flags |= cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
        self.get_per_plane_correspondences(mode=mode, use_ransac=use_ransac)
        if len(self.obj_pts) == 0:
            return None, None
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.obj_pts, self.img_pts, (self.image_width, self.image_height), None, None, flags=flags
            )
        except cv2.error:
            return None, None
        if not ret:
            return None, None
        self.calibration = mtx
        rot, _ = cv2.Rodrigues(rvecs[0])
        self.rotation = rot
        self.position = (-np.transpose(self.rotation) @ tvecs[0]).T[0]
        if self.ord_pts[0] != 0:
            self.change_plane_coords()
        pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(self.rotation)
        return {
            "pan_degrees": float(np.rad2deg(pan)),
            "tilt_degrees": float(np.rad2deg(tilt)),
            "roll_degrees": float(np.rad2deg(roll)),
            "x_focal_length": float(mtx[0, 0]),
            "y_focal_length": float(mtx[1, 1]),
            "principal_point": [self.image_width / 2.0, self.image_height / 2.0],
            "position_meters": [float(x) for x in self.position],
            "rotation_matrix": self.rotation.tolist(),
            "radial_distortion": [float(dist.ravel()[0]), 0.0, 0.0, 0.0, 0.0, 0.0],
            "tangential_distortion": [0.0, 0.0],
            "thin_prism_distortion": [0.0, 0.0, 0.0, 0.0],
        }, ret


def decode_keypoints(kp_arr, line_arr, device):
    kp = torch.from_numpy(kp_arr.astype(np.float32)).unsqueeze(0).to(device)
    line = torch.from_numpy(line_arr.astype(np.float32)).unsqueeze(0).to(device)
    kc = get_keypoints_from_heatmap_batch_maxpool(kp[:, :-1])
    lc = get_keypoints_from_heatmap_batch_maxpool_l(line[:, :-1])
    return complete_keypoints(
        coords_to_dict(kc, threshold=0.1449),
        coords_to_dict(lc, threshold=0.2983),
        w=960,
        h=540,
        normalize=True,
    )[0]


def solve_params(keypoints, radial=False):
    cls = RadialK1Calib if radial else FramebyFrameCalib
    cam = cls(1920, 1080, denormalize=True)
    cam.update(copy.deepcopy(keypoints))
    h = cam.get_homography_from_ground_plane(use_ransac=50, inverse=True)
    if h is None:
        return {}
    try:
        vr = cam.heuristic_voting()
        return vr["cam_params"] if vr else {}
    except Exception:
        return {}


def tripod_gate(params, history, stats):
    if not params or len(history) < 8:
        if params:
            history.append(np.array(params["position_meters"], dtype=float))
        return params
    cur = np.array(params["position_meters"], dtype=float)
    arr = np.stack(history)
    med = np.median(arr, axis=0)
    dist = float(np.linalg.norm(cur - med))
    mad = float(np.median(np.linalg.norm(arr - med, axis=1)))
    if dist > max(8.0, 4.0 * mad):
        out = dict(params)
        out["position_meters"] = [float(x) for x in med]
        history.append(med)
        stats["tripod_replaced"] += 1
        return out
    history.append(cur)
    return params


def image_path_for_frame(vid, frame):
    idx = int(frame) % 1000000
    return f"{DATA_ROOT}/SNGS-{vid}/img1/{idx:06d}.jpg"


def flow_outlier_replace(base, prev_gray, cur_gray, prev_kp, stats):
    if prev_gray is None or prev_kp is None:
        return base
    outliers = set(base) - ransac_inlier_keys(base)
    keys = [k for k in outliers if k in prev_kp]
    if not keys:
        return base
    p0 = np.array([[prev_kp[k]["x"] * 1920.0, prev_kp[k]["y"] * 1080.0] for k in keys], dtype=np.float32).reshape(-1, 1, 2)
    p1, st, _err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None, winSize=(21, 21), maxLevel=3)
    merged = copy.deepcopy(base)
    for key, point, ok in zip(keys, p1.reshape(-1, 2), st.reshape(-1)):
        if not ok:
            continue
        x, y = float(point[0] / 1920.0), float(point[1] / 1080.0)
        if -0.1 <= x <= 1.1 and -0.1 <= y <= 1.1:
            merged[key]["x"], merged[key]["y"] = x, y
            stats["flow_replaced"] += 1
    still = set(merged) - ransac_inlier_keys(merged)
    for key in still:
        merged.pop(key, None)
    stats["flow_removed"] += len(still)
    stats["flow_initial_outliers"] += len(outliers)
    return merged


def solve_video(args):
    vid, method, cache_root, stride, device, max_frames, progress_dir = args
    files = sorted(glob.glob(f"{cache_root}/test/SNGS-{vid}/frame_*.npz"))
    out, stats = {}, defaultdict(int)
    pos_history = deque(maxlen=30)
    prev_gray, prev_kp = None, None
    scored = 0
    progress_path = Path(progress_dir) / f"{method}_SNGS-{vid}.json" if progress_dir else None
    for i, path in enumerate(files):
        if i % stride != 0:
            continue
        try:
            d = np.load(path)
            frame = int(d["frame"])
            base = decode_keypoints(d["kp_hm"], d["line_hm"], device)
        except Exception:
            stats["bad_cache"] += 1
            continue
        cur_gray = None
        if "flow" in method:
            img = cv2.imread(image_path_for_frame(vid, frame), cv2.IMREAD_GRAYSCALE)
            cur_gray = img
        if i % stride == 0:
            pred = base
            if "flow" in method and cur_gray is not None:
                pred = flow_outlier_replace(base, prev_gray, cur_gray, prev_kp, stats)
            params = solve_params(pred, radial=("radial" in method))
            if "tripod" in method:
                params = tripod_gate(params, pos_history, stats)
            out[f"3{vid}{frame % 1000000:06d}"] = params
            scored += 1
            if progress_path:
                stats["last_frame"] = frame
                stats["scored"] = scored
                progress_path.write_text(
                    json.dumps({"method": method, "video": vid, "frames": len(out), "stats": dict(stats)}, indent=2),
                    encoding="utf-8",
                )
            if max_frames and scored >= max_frames:
                break
        if cur_gray is not None:
            prev_gray = cur_gray
            prev_kp = base
            stats["flow_scored_only"] = 1
    return method, vid, out, smoothness(out), dict(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--videos", nargs="+", default=[str(x) for x in list(range(116, 151)) + list(range(187, 201))])
    ap.add_argument("--max-frames", type=int, default=0, help="Max scored frames per video/method; 0 means all.")
    ap.add_argument("--progress-dir", default=None, help="Directory for per-video progress JSON files.")
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "radial_k1", "tripod", "flow", "flow_tripod", "radial_tripod"],
    )
    args = ap.parse_args()
    if args.stride < 20:
        raise SystemExit("--stride must be >= 20 for comparable calibration eval")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = Path(args.progress_dir) if args.progress_dir else out_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (v, m, args.cache_root, args.stride, args.device, args.max_frames, str(progress_dir))
        for m in args.methods
        for v in args.videos
    ]
    params = {m: {} for m in args.methods}
    sm = {m: [] for m in args.methods}
    aux = {m: defaultdict(int) for m in args.methods}
    with Pool(min(args.nproc, len(jobs))) as pool:
        for method, vid, pb, s, st in pool.imap_unordered(solve_video, jobs):
            params[method][vid] = pb
            sm[method].append(s)
            for k, val in st.items():
                aux[method][k] += val
            print(f"params method={method} video={vid} frames={len(pb)}", flush=True)
    json.dump(params, open(out_dir / "params.json", "w"))
    score_jobs = [(vid, method, params[method][vid]) for method in params for vid in params[method]]
    with Pool(args.nproc) as pool:
        parts = pool.map(_score_job, score_jobs)
    agg = {m: {"micro": [], "macro": [], "reproj": [], "nvid": 0} for m in args.methods}
    for method, _vid, mi, ma, rj in parts:
        agg[method]["micro"] += mi
        agg[method]["macro"] += ma
        agg[method]["reproj"] += rj
        agg[method]["nvid"] += 1
    res = {}
    for method, g in agg.items():
        mean_sm = [x["mean"] for x in sm[method] if x["mean"] is not None]
        p95_sm = [x["p95"] for x in sm[method] if x["p95"] is not None]
        res[method] = {
            "point_acc": float(np.mean(g["micro"])) if g["micro"] else None,
            "line_acc": float(np.mean(g["macro"])) if g["macro"] else None,
            "reproj_mean": float(np.mean(g["reproj"])) if g["reproj"] else None,
            "n_frames": len(g["micro"]),
            "n_videos": g["nvid"],
            "smoothness_mean": float(np.mean(mean_sm)) if mean_sm else None,
            "smoothness_p95": float(np.mean(p95_sm)) if p95_sm else None,
            "stats": dict(aux[method]),
        }
    json.dump(res, open(out_dir / "result.json", "w"), indent=2)

    def fmt(x, n=4):
        return "NA" if x is None else f"{x:.{n}f}"

    lines = [
        "| method | point | line | reproj | smooth_mean | smooth_p95 | frames | stats |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for method, r in sorted(res.items()):
        lines.append(
            f"| {method} | {fmt(r['point_acc'])} | {fmt(r['line_acc'])} | {fmt(r['reproj_mean'], 2)} | "
            f"{fmt(r['smoothness_mean'], 2)} | {fmt(r['smoothness_p95'], 2)} | {r['n_frames']} | "
            f"`{json.dumps(r['stats'], sort_keys=True)}` |"
        )
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((out_dir / "RESULTS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
