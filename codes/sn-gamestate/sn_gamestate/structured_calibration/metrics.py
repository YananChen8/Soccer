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
DEFAULT_THRESHOLD = 10
THRESHOLD = DEFAULT_THRESHOLD
PARAM_FIELDS_SCALAR = ["x_focal_length", "y_focal_length", "pan_degrees",
                       "tilt_degrees", "roll_degrees"]

B2P_TEMPLATE_TO_SOCCER = np.array([
    [1.0, 0.0, -62.5],
    [0.0, 1.0, -39.0],
    [0.0, 0.0, 1.0],
], dtype=float)
CENTRAL_CIRCLE_RADIUS = 9.14
HOMOGRAPHY_LINE_ENDPOINTS_TEMPLATE = {
    "Big rect. left top": [[10.0, 18.84], [26.50, 18.84]],
    "Big rect. left bottom": [[10.0, 59.16], [26.50, 59.16]],
    "Big rect. left main": [[26.50, 59.16], [26.50, 18.84]],
    "Big rect. right top": [[98.50, 18.84], [115.0, 18.84]],
    "Big rect. right bottom": [[98.50, 59.16], [115.0, 59.16]],
    "Big rect. right main": [[98.50, 59.16], [98.50, 18.84]],
    "Small rect. left top": [[10.0, 29.84], [15.5, 29.84]],
    "Small rect. left bottom": [[10.0, 48.16], [15.5, 48.16]],
    "Small rect. left main": [[15.5, 48.16], [15.5, 29.84]],
    "Small rect. right top": [[109.50, 29.84], [115.0, 29.84]],
    "Small rect. right bottom": [[109.50, 48.16], [115.0, 48.16]],
    "Small rect. right main": [[109.50, 48.16], [109.50, 29.84]],
    "Side line bottom": [[10.0, 73.0], [115.0, 73.0]],
    "Side line top": [[10.0, 5.0], [115.0, 5.0]],
    "Side line left": [[10.0, 73.0], [10.0, 5.0]],
    "Side line right": [[115.0, 73.0], [115.0, 5.0]],
    "Middle line": [[62.50, 73.0], [62.50, 5.0]],
}

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


def _video_dir_name(vid) -> str:
    text = str(vid)
    if text.upper().startswith("SNGS-"):
        return text
    return f"SNGS-{text}"


def load_gt_lines_for_video(data_root: Path, vid: str):
    p = Path(data_root) / _video_dir_name(vid) / "Labels-GameState.json"
    d = json.load(open(p))
    out = {}
    for ann in d["annotations"]:
        if ann.get("supercategory") != "pitch":
            continue
        out[str(ann["image_id"])] = scale_points(ann["lines"], WIDTH, HEIGHT)
    return out


def _safe_mirror_eval(pred_lines, gt_lines, threshold=THRESHOLD):
    """evaluate_camera_prediction against mirrored GT, robust to GT classes that
    are absent from SoccerPitch.symetric_classes (a known SoccerNet GT artifact,
    e.g. 'Goal left post left')."""
    try:
        return evaluate_camera_prediction(pred_lines, mirror_labels(gt_lines), threshold)
    except KeyError:
        return None, None, None


def _eval_one_video(vid, pred_by_id, data_root, stride=1, threshold=THRESHOLD):
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
        c1, _, r1 = evaluate_camera_prediction(pred_lines, gt_lines, threshold)
        a1 = c1[0, 0] / c1.sum() if c1.sum() > 0 else 0.0
        c2, _, r2 = _safe_mirror_eval(pred_lines, gt_lines, threshold)
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


def accuracy_eval(params_by_imgid_per_video: dict, data_root, videos, nproc=10, stride=1, threshold=THRESHOLD):
    """params_by_imgid_per_video: {vid: {image_id(str): params_dict}}.
    Parallelized across videos (get_polylines is CPU-heavy). `stride` subsamples
    frames for fast iteration (accuracy is a per-frame mean, robust to subsampling;
    jitter metrics always use the full sequence, computed separately)."""
    jobs = [(vid, params_by_imgid_per_video.get(vid, {}), str(data_root), stride, threshold) for vid in videos]
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


def _per_frame_acc_one_video(vid, pred_by_id, data_root, stride=1, threshold=THRESHOLD):
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
        c1, _, _ = evaluate_camera_prediction(pred_lines, gt_lines, threshold)
        a1 = c1[0, 0] / c1.sum() if c1.sum() > 0 else 0.0
        c2, _, _ = _safe_mirror_eval(pred_lines, gt_lines, threshold)
        a2 = (c2[0, 0] / c2.sum() if c2.sum() > 0 else 0.0) if c2 is not None else -1.0
        out[image_id] = max(a1, a2)
    return out


def per_frame_accuracy(params_by_imgid_per_video: dict, data_root, videos, nproc=10, stride=1, threshold=THRESHOLD):
    """Returns {vid: {image_id: acc or None}} so callers can subset frames."""
    jobs = [(vid, params_by_imgid_per_video.get(vid, {}), str(data_root), stride, threshold) for vid in videos]
    if nproc and nproc > 1:
        with Pool(min(nproc, len(jobs))) as pool:
            parts = pool.starmap(_per_frame_acc_one_video, jobs)
    else:
        parts = [_per_frame_acc_one_video(*j) for j in jobs]
    return {vid: part for vid, part in zip(videos, parts)}


def _normalize_homography(h):
    if h is None:
        return None
    try:
        arr = np.asarray(h, dtype=float)
    except Exception:
        return None
    if arr.shape != (3, 3) or not np.all(np.isfinite(arr)) or abs(arr[2, 2]) < 1e-12:
        return None
    return arr / arr[2, 2]


def _project_points_h(h, points_xy):
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    pts_h = np.concatenate([pts[:, :2], np.ones((len(pts), 1), dtype=float)], axis=1)
    proj = (h @ pts_h.T).T
    denom = proj[:, 2:3]
    valid = np.isfinite(proj).all(axis=1) & (np.abs(denom[:, 0]) > 1e-12)
    out = np.full((len(pts), 2), np.nan, dtype=float)
    out[valid] = proj[valid, :2] / denom[valid]
    return out, valid


def _template_to_soccer(points_xy):
    out, valid = _project_points_h(B2P_TEMPLATE_TO_SOCCER, points_xy)
    if not np.all(valid):
        raise ValueError("invalid template-to-soccer conversion")
    return out


def _line_sample(points_xy, num=80):
    p = np.asarray(points_xy, dtype=float)
    t = np.linspace(0.0, 1.0, num)
    return p[0][None, :] * (1.0 - t[:, None]) + p[1][None, :] * t[:, None]


def _circle_sample(radius=CENTRAL_CIRCLE_RADIUS, num=120):
    theta = np.linspace(0.0, 2.0 * np.pi, num)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)


def homography_polylines(h_image_to_pitch, width=WIDTH, height=HEIGHT, sampling_factor=0.9):
    """Project standard pitch markings into the image from an image->pitch H.

    The returned dict has the same point-list shape as SoccerNet line
    annotations. This is a direct Homography evaluator path for states that do
    not carry camera parameter dictionaries.
    """
    h = _normalize_homography(h_image_to_pitch)
    if h is None:
        raise ValueError("invalid homography")
    h_pitch_to_image = np.linalg.inv(h)
    lines = {}
    samples_per_line = max(8, int(80 * float(sampling_factor)))
    for name, endpoints in HOMOGRAPHY_LINE_ENDPOINTS_TEMPLATE.items():
        soccer = _template_to_soccer(_line_sample(endpoints, samples_per_line))
        image_pts, valid = _project_points_h(h_pitch_to_image, soccer)
        pts = image_pts[valid]
        if len(pts) >= 2:
            lines[name] = [{"x": float(x), "y": float(y)} for x, y in pts]
    circle_soccer = _circle_sample(num=max(24, int(120 * float(sampling_factor))))
    circle_img, valid = _project_points_h(h_pitch_to_image, circle_soccer)
    circle_pts = circle_img[valid]
    if len(circle_pts) >= 8:
        lines["Circle central"] = [{"x": float(x), "y": float(y)} for x, y in circle_pts]
    return lines


def _points_from_line_payload(points):
    out = []
    if points is None:
        return np.zeros((0, 2), dtype=float)
    for point in points:
        if isinstance(point, dict) and "x" in point and "y" in point:
            out.append([float(point["x"]), float(point["y"])])
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            out.append([float(point[0]), float(point[1])])
    return np.asarray(out, dtype=float)


def _nearest_distances(src, dst):
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if len(src) == 0 or len(dst) == 0:
        return np.full((len(src),), np.inf, dtype=float)
    d = np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=2)
    return d.min(axis=1)


def evaluate_homography_line_prediction(pred_lines, gt_lines, threshold=5.0):
    """Compare projected Homography lines with GT line annotations at threshold.

    The score is intentionally named generically. Use ``threshold=5`` when a
    JaC@5-style pixel gate is desired; do not report 10px runs as JaC@5.
    """
    confusion = np.zeros((2, 2), dtype=float)
    reproj = {}
    for name, gt_payload in gt_lines.items():
        gt_pts = _points_from_line_payload(gt_payload)
        pred_pts = _points_from_line_payload(pred_lines.get(name, []))
        if len(gt_pts) == 0:
            continue
        dists = _nearest_distances(gt_pts, pred_pts)
        reproj[name] = [float(d) for d in dists if np.isfinite(d)]
        confusion[0, 0] += float(np.sum(dists <= threshold))
        confusion[1, 0] += float(np.sum(dists > threshold))
    for name, pred_payload in pred_lines.items():
        pred_pts = _points_from_line_payload(pred_payload)
        gt_pts = _points_from_line_payload(gt_lines.get(name, []))
        if len(pred_pts) == 0:
            continue
        dists = _nearest_distances(pred_pts, gt_pts)
        confusion[0, 1] += float(np.sum(dists > threshold))
    denom = confusion.sum()
    accuracy = confusion[0, 0] / denom if denom > 0 else 0.0
    return confusion, reproj, accuracy


def _homography_eval_one_video(vid, pred_by_id, data_root, stride=1, threshold=5.0):
    gt = load_gt_lines_for_video(data_root, vid)
    items = sorted(gt.items(), key=lambda kv: int(kv[0]))
    if stride > 1:
        items = items[::stride]
    v_acc, prec, rec, reproj_all = [], [], [], []
    total, missed = 0, 0
    for image_id, gt_lines in items:
        total += 1
        h = pred_by_id.get(image_id)
        try:
            pred_lines = homography_polylines(h, WIDTH, HEIGHT, sampling_factor=0.9)
        except Exception:
            missed += 1
            continue
        confusion, reproj, acc = evaluate_homography_line_prediction(pred_lines, gt_lines, threshold)
        v_acc.append(acc)
        if confusion[0, :].sum() > 0:
            prec.append(confusion[0, 0] / confusion[0, :].sum())
        if (confusion[0, 0] + confusion[1, 0]) > 0:
            rec.append(confusion[0, 0] / (confusion[0, 0] + confusion[1, 0]))
        for errs in reproj.values():
            reproj_all.extend(errs)
    return {"vid": vid, "total": total, "missed": missed, "acc": v_acc,
            "prec": prec, "rec": rec, "reproj": reproj_all}


def homography_accuracy_eval(h_by_imgid_per_video: dict, data_root, videos, nproc=10, stride=1, threshold=5.0):
    """Evaluate image->pitch Homographies directly against GT line annotations."""
    jobs = [(vid, h_by_imgid_per_video.get(vid, {}), str(data_root), stride, threshold) for vid in videos]
    if nproc and nproc > 1:
        with Pool(min(nproc, len(jobs))) as pool:
            parts = pool.starmap(_homography_eval_one_video, jobs)
    else:
        parts = [_homography_eval_one_video(*j) for j in jobs]

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
        "threshold_px": float(threshold),
        "completeness": (total - missed) / total if total else None,
        "meanAccuracy": float(np.mean(accuracies)) if accuracies else None,
        "meanPrecision": float(np.mean(precisions)) if precisions else None,
        "meanRecall": float(np.mean(recalls)) if recalls else None,
        "reproj_mean_px": float(np.mean(reproj_all)) if reproj_all else None,
        "reproj_median_px": float(np.median(reproj_all)) if reproj_all else None,
        "per_video": per_video,
    }


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
