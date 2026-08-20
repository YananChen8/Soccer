"""Metrics for calibration post-processing evaluation.

Three families:
  * accuracy   -- official SoccerNet line-reprojection (needs line GT) from `parameters`
  * param_jitter -- frame-to-frame jitter of pan/tilt/roll/focal/position (GT-free)
  * field_anchor_jitter -- frame-to-frame pixel movement of fixed 3D pitch anchors
                           projected into the image (GT-free; proxies overlay/BEV stability)

All operate on the camera `parameters` dict, so they apply identically to raw and
stabilized outputs.
"""
from __future__ import annotations

import json
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from SoccerNet.Evaluation.utils_calibration import (
    Camera, SoccerPitch, get_polylines, scale_points,
    evaluate_camera_prediction, mirror_labels,
)

WIDTH, HEIGHT = 1920, 1080
THRESHOLD = 10
PARAM_FIELDS_SCALAR = ["x_focal_length", "y_focal_length", "pan_degrees",
                       "tilt_degrees", "roll_degrees"]

# Fixed, well-spread 3D anchors for projection-stability measurement.
_ANCHOR_NAMES = [
    "center_mark",
    "top_left_corner", "top_right_corner", "bottom_left_corner", "bottom_right_corner",
    "left_penalty_mark", "right_penalty_mark",
    "halfway_and_top_touch_line_mark", "halfway_and_bottom_touch_line_mark",
]


def _anchor_points():
    sp = SoccerPitch()
    pts = {}
    for name in _ANCHOR_NAMES:
        p = getattr(sp, name, None)
        if p is not None:
            pts[name] = np.array(p, dtype=float)
    return pts


_ANCHORS = _anchor_points()


def load_gt_lines_for_video(data_root: Path, vid: str):
    p = Path(data_root) / f"SNGS-{vid}" / "Labels-GameState.json"
    d = json.load(open(p))
    out = {}
    for ann in d["annotations"]:
        if ann.get("supercategory") != "pitch":
            continue
        out[str(ann["image_id"])] = scale_points(ann["lines"], WIDTH, HEIGHT)
    return out


def _safe_mirror_eval(pred_lines, gt_lines):
    """evaluate_camera_prediction against mirrored GT, robust to GT classes that
    are absent from SoccerPitch.symetric_classes (a known SoccerNet GT artifact,
    e.g. 'Goal left post left')."""
    try:
        return evaluate_camera_prediction(pred_lines, mirror_labels(gt_lines), THRESHOLD)
    except KeyError:
        return None, None, None


def _eval_one_video(vid, pred_by_id, data_root, stride=1):
    gt = load_gt_lines_for_video(data_root, vid)
    # frame order is encoded in image_id; stride subsamples for fast iteration
    items = sorted(gt.items(), key=lambda kv: int(kv[0]))
    if stride > 1:
        items = items[::stride]
    v_acc, prec, rec, reproj_all = [], [], [], []
    total, missed = 0, 0
    for image_id, gt_lines in items:
        total += 1
        params = pred_by_id.get(image_id)
        if not isinstance(params, dict) or len(params) == 0:
            missed += 1
            continue
        try:
            pred_lines = get_polylines(params, WIDTH, HEIGHT, sampling_factor=0.9)
        except Exception:
            missed += 1
            continue
        c1, _, r1 = evaluate_camera_prediction(pred_lines, gt_lines, THRESHOLD)
        a1 = c1[0, 0] / c1.sum() if c1.sum() > 0 else 0.0
        c2, _, r2 = _safe_mirror_eval(pred_lines, gt_lines)
        a2 = (c2[0, 0] / c2.sum() if c2.sum() > 0 else 0.0) if c2 is not None else -1.0
        confusion, reproj, acc = (c1, r1, a1) if a1 >= a2 else (c2, r2, a2)
        v_acc.append(acc)
        if confusion[0, :].sum() > 0:
            prec.append(confusion[0, 0] / confusion[0, :].sum())
        if (confusion[0, 0] + confusion[1, 0]) > 0:
            rec.append(confusion[0, 0] / (confusion[0, 0] + confusion[1, 0]))
        for errs in reproj.values():
            reproj_all.extend(errs)
    return {"vid": vid, "total": total, "missed": missed, "acc": v_acc,
            "prec": prec, "rec": rec, "reproj": reproj_all}


def accuracy_eval(params_by_imgid_per_video: dict, data_root, videos, nproc=10, stride=1):
    """params_by_imgid_per_video: {vid: {image_id(str): params_dict}}.
    Parallelized across videos (get_polylines is CPU-heavy). `stride` subsamples
    frames for fast iteration (accuracy is a per-frame mean, robust to subsampling;
    jitter metrics always use the full sequence, computed separately)."""
    jobs = [(vid, params_by_imgid_per_video.get(vid, {}), str(data_root), stride) for vid in videos]
    if nproc and nproc > 1:
        with Pool(min(nproc, len(jobs))) as pool:
            parts = pool.starmap(_eval_one_video, jobs)
    else:
        parts = [_eval_one_video(*j) for j in jobs]

    accuracies, precisions, recalls, reproj_all = [], [], [], []
    total, missed, per_video = 0, 0, {}
    for r in parts:
        total += r["total"]; missed += r["missed"]
        accuracies += r["acc"]; precisions += r["prec"]; recalls += r["rec"]
        reproj_all += r["reproj"]
        per_video[r["vid"]] = {
            "n_matched": len(r["acc"]),
            "meanAccuracy": float(np.mean(r["acc"])) if r["acc"] else None,
        }
    return {
        "completeness": (total - missed) / total if total else None,
        "meanAccuracy": float(np.mean(accuracies)) if accuracies else None,
        "meanPrecision": float(np.mean(precisions)) if precisions else None,
        "meanRecall": float(np.mean(recalls)) if recalls else None,
        "reproj_mean_px": float(np.mean(reproj_all)) if reproj_all else None,
        "reproj_median_px": float(np.median(reproj_all)) if reproj_all else None,
        "per_video": per_video,
    }


def _self_consist_one_video(vid, frame_lines_params):
    """frame_lines_params: list of (image_id, params_dict, detected_lines_norm).
    Returns {image_id: reproj_err_px or None}. err = median over detected line
    points of nearest distance to the field model reprojected by `params`.
    GT-free: measures whether the solved camera explains its own observations."""
    out = {}
    for image_id, params, det_lines in frame_lines_params:
        if not (isinstance(params, dict) and len(params)) or not det_lines:
            out[image_id] = None
            continue
        try:
            pred = get_polylines(params, WIDTH, HEIGHT, sampling_factor=0.9)
        except Exception:
            out[image_id] = None
            continue
        dists = []
        for cls, det_pts in det_lines.items():
            pred_pts = pred.get(cls)
            if not pred_pts or not det_pts:
                continue
            P = np.array([[p["x"], p["y"]] for p in pred_pts], dtype=float)
            for dp in det_pts:
                d = np.array([dp["x"] * WIDTH, dp["y"] * HEIGHT])  # denormalize
                dists.append(float(np.min(np.linalg.norm(P - d, axis=1))))
        out[image_id] = float(np.median(dists)) if dists else None
    return out


def self_consistency(frame_lines_params_per_video: dict, videos, nproc=10):
    """{vid: [(image_id, params, detected_lines_norm), ...]} -> {vid: {image_id: err_px}}."""
    jobs = [(vid, frame_lines_params_per_video.get(vid, [])) for vid in videos]
    if nproc and nproc > 1:
        with Pool(min(nproc, len(jobs))) as pool:
            parts = pool.starmap(_self_consist_one_video, jobs)
    else:
        parts = [_self_consist_one_video(*j) for j in jobs]
    return {vid: part for vid, part in zip(videos, parts)}


def _per_frame_acc_one_video(vid, pred_by_id, data_root, stride=1):
    gt = load_gt_lines_for_video(data_root, vid)
    items = sorted(gt.items(), key=lambda kv: int(kv[0]))
    if stride > 1:
        items = items[::stride]
    out = {}
    for image_id, gt_lines in items:
        params = pred_by_id.get(image_id)
        if not isinstance(params, dict) or len(params) == 0:
            out[image_id] = None
            continue
        try:
            pred_lines = get_polylines(params, WIDTH, HEIGHT, sampling_factor=0.9)
        except Exception:
            out[image_id] = None
            continue
        c1, _, _ = evaluate_camera_prediction(pred_lines, gt_lines, THRESHOLD)
        a1 = c1[0, 0] / c1.sum() if c1.sum() > 0 else 0.0
        c2, _, _ = _safe_mirror_eval(pred_lines, gt_lines)
        a2 = (c2[0, 0] / c2.sum() if c2.sum() > 0 else 0.0) if c2 is not None else -1.0
        out[image_id] = max(a1, a2)
    return out


def per_frame_accuracy(params_by_imgid_per_video: dict, data_root, videos, nproc=10, stride=1):
    """Returns {vid: {image_id: acc or None}} so callers can subset frames."""
    jobs = [(vid, params_by_imgid_per_video.get(vid, {}), str(data_root), stride) for vid in videos]
    if nproc and nproc > 1:
        with Pool(min(nproc, len(jobs))) as pool:
            parts = pool.starmap(_per_frame_acc_one_video, jobs)
    else:
        parts = [_per_frame_acc_one_video(*j) for j in jobs]
    return {vid: part for vid, part in zip(videos, parts)}


def _series_jitter(arr: np.ndarray):
    diffs = np.diff(arr)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return None
    jerk = np.diff(diffs)
    return {
        "mean_abs_diff": float(np.mean(np.abs(diffs))),
        "max_abs_diff": float(np.max(np.abs(diffs))),
        "total_variation": float(np.sum(np.abs(diffs))),
        "rms_jerk": float(np.sqrt(np.mean(jerk ** 2))) if len(jerk) else None,
    }


def param_jitter(rows_per_video: dict, videos):
    """rows_per_video: {vid: [(frame, image_id, params_dict_or_None), ...]} ordered."""
    agg = defaultdict(list)
    per_video = {}
    for vid in videos:
        rows = rows_per_video.get(vid, [])
        series = defaultdict(list)
        for _, _, params in rows:
            ok = isinstance(params, dict) and len(params)
            for f in PARAM_FIELDS_SCALAR:
                series[f].append(float(params[f]) if ok and f in params else np.nan)
            pos = params.get("position_meters", [np.nan] * 3) if ok else [np.nan] * 3
            for i, ax in enumerate(["x", "y", "z"]):
                series[f"position_meters_{ax}"].append(pos[i] if len(pos) > i else np.nan)
        vstats = {}
        for fldname, vals in series.items():
            st = _series_jitter(np.array(vals, dtype=float))
            if st:
                vstats[fldname] = st
                for k, v in st.items():
                    if v is not None:
                        agg[f"{fldname}.{k}"].append(v)
        per_video[vid] = vstats
    aggregate = {k: float(np.mean(v)) for k, v in agg.items() if v}
    return {"aggregate": aggregate, "per_video": per_video}


def field_anchor_jitter(rows_per_video: dict, videos):
    """Mean per-anchor frame-to-frame pixel movement, only counting anchors whose
    projection is in-frame (z>0, inside a generous image margin) in BOTH frames."""
    agg = []
    per_video = {}
    margin = 0.5  # allow anchors up to half a frame outside before discarding
    xmin, xmax = -margin * WIDTH, (1 + margin) * WIDTH
    ymin, ymax = -margin * HEIGHT, (1 + margin) * HEIGHT
    for vid in videos:
        rows = rows_per_video.get(vid, [])
        proj_seq = []  # list of {anchor: (x,y) or None}
        for _, _, params in rows:
            if not (isinstance(params, dict) and len(params)):
                proj_seq.append({})
                continue
            cam = Camera(WIDTH, HEIGHT)
            try:
                cam.from_json_parameters(params)
            except Exception:
                proj_seq.append({})
                continue
            frame_proj = {}
            for name, p3 in _ANCHORS.items():
                xy = cam.project_point(p3)
                if xy[2] != 0 and xmin <= xy[0] <= xmax and ymin <= xy[1] <= ymax:
                    frame_proj[name] = np.array([xy[0], xy[1]])
            proj_seq.append(frame_proj)
        moves = []
        for a, b in zip(proj_seq[:-1], proj_seq[1:]):
            for name in a.keys() & b.keys():
                moves.append(float(np.linalg.norm(a[name] - b[name])))
        v_mean = float(np.mean(moves)) if moves else None
        per_video[vid] = {"mean_px_move": v_mean, "n_pairs": len(moves)}
        if v_mean is not None:
            agg.append(v_mean)
    return {"mean_px_move": float(np.mean(agg)) if agg else None, "per_video": per_video}
