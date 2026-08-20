"""Field-graph primitives (innovation #1), v1: GoalFrame.

v1 operates by re-solving the camera from the saved keypoints with the base
NBJW solver (FramebyFrameCalib.heuristic_voting), optionally suppressing the
goal-frame keypoint group. The goal points {12..19} include the non-planar
crossbar/post-top corners {12,15,16,19} (zw=-2.44m) which, when mislocalized,
strongly pollute the 'full'/'main' camera solve.

GoalFramePrimitive policy choices (selectable):
  - keep      : use all keypoints (= baseline)
  - suppress  : always drop the goal group, re-solve
  - gated     : drop the goal group only when it is *influential* (including it
                vs excluding it disagrees beyond a threshold) -- i.e. only when
                the goal points actually move the camera, prefer the
                better-spread goal-free solution.
"""
from __future__ import annotations

import copy
import sys

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

# elevated (non-planar) goal corners + goal-post bottoms, per nbjw kp_to_line
GOAL_ELEVATED = {12, 15, 16, 19}
GOAL_ALL = {12, 13, 14, 15, 16, 17, 18, 19}

# center-circle keypoint indices (Circle central/left/right) and middle line, per kp_to_line
CIRCLE_IDX = {32, 48, 38, 50, 42, 53, 35, 54, 43, 52, 39, 49,
              31, 37, 47, 41, 34, 33, 40, 55, 44, 36}
MIDDLE_IDX = {2, 51, 29}  # middle-line-only points (32,35 are circle∩middle, snapped to ellipse)


def ensure_nbjw_on_path(project_sn_gamestate):
    p1 = f"{project_sn_gamestate}/plugins/calibration/nbjw_calib"
    p2 = f"{project_sn_gamestate}/plugins/calibration"
    for p in (p1, p2):
        if p not in sys.path:
            sys.path.insert(0, p)


def _solve(FramebyFrameCalib, kps, drop=()):
    cam = FramebyFrameCalib(1920, 1080, denormalize=False)
    d = {k: {'x': v['x'], 'y': v['y']} for k, v in kps.items() if k not in drop}
    cam.update(d)
    r = cam.heuristic_voting()
    return r['cam_params'] if r else None


def _pose_delta(a, b):
    """Rough pose disagreement: max of |Δpan|,|Δtilt|,|Δroll| (deg) and 100*|Δf|/f."""
    if a is None or b is None:
        return float('inf')
    dang = max(abs(a['pan_degrees'] - b['pan_degrees']),
               abs(a['tilt_degrees'] - b['tilt_degrees']),
               abs(a['roll_degrees'] - b['roll_degrees']))
    fa, fb = a['x_focal_length'], b['x_focal_length']
    dfoc = 100.0 * abs(fa - fb) / max(abs(fa), 1e-6)
    return max(dang, dfoc)


def _ellipse_points(center, axes, angle_deg, n=720):
    a, b = axes[0] / 2.0, axes[1] / 2.0
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = a * np.cos(th); y = b * np.sin(th)
    ca, sa = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    X = center[0] + x * ca - y * sa
    Y = center[1] + x * sa + y * ca
    return np.stack([X, Y], axis=1)


def _snap_to_curve(pt, curve_pts):
    d = np.linalg.norm(curve_pts - np.array(pt), axis=1)
    return curve_pts[int(np.argmin(d))]


def _fit_line_tls(pts):
    """Total-least-squares line fit; returns (point_on_line, unit_dir)."""
    P = np.asarray(pts, dtype=float)
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c)
    return c, vt[0]


def centercircle_correct(kps, min_circle_pts=5, max_snap_px=40.0):
    """Denoise center-circle & middle-line keypoints by snapping them to a fitted
    ellipse / line (the doc's 'Correction' op). Returns (new_kps, info).

    Only snaps a point if it lies within max_snap_px of the fitted structure (so a
    grossly wrong detection is left for the solver's RANSAC rather than dragged
    onto a fit it corrupted). No-op when too few circle points are present."""
    info = {"n_circle": 0, "n_middle": 0, "snapped_circle": 0, "snapped_middle": 0, "fit": False}
    out = copy.deepcopy(kps)
    circ = [(k, kps[k]) for k in kps if k in CIRCLE_IDX]
    info["n_circle"] = len(circ)
    if cv2 is None or len(circ) < min_circle_pts:
        return out, info
    pts = np.array([[v['x'], v['y']] for _k, v in circ], dtype=np.float32)
    try:
        (cx, cy), (MA, ma), ang = cv2.fitEllipse(pts)
    except Exception:
        return out, info
    if not (np.isfinite(cx) and np.isfinite(cy) and MA > 1 and ma > 1):
        return out, info
    info["fit"] = True
    ell = _ellipse_points((cx, cy), (MA, ma), ang)
    for k, v in circ:
        sp = _snap_to_curve((v['x'], v['y']), ell)
        if np.linalg.norm(sp - np.array([v['x'], v['y']])) <= max_snap_px:
            out[k] = {**v, 'x': float(sp[0]), 'y': float(sp[1])}
            info["snapped_circle"] += 1
    # middle line: fit from middle-only points (+ circle∩middle 32,35 if present)
    mid_keys = [k for k in kps if k in MIDDLE_IDX]
    anchor = [k for k in (32, 35) if k in kps]
    fit_keys = mid_keys + anchor
    info["n_middle"] = len(mid_keys)
    if len(fit_keys) >= 2 and len(mid_keys) >= 1:
        lp = np.array([[kps[k]['x'], kps[k]['y']] for k in fit_keys], dtype=float)
        c, d = _fit_line_tls(lp)
        for k in mid_keys:
            p = np.array([kps[k]['x'], kps[k]['y']])
            proj = c + np.dot(p - c, d) * d
            if np.linalg.norm(proj - p) <= max_snap_px:
                out[k] = {**kps[k], 'x': float(proj[0]), 'y': float(proj[1])}
                info["snapped_middle"] += 1
    return out, info


def centercircle_resolve(FramebyFrameCalib, kps, saved_params, **kw):
    new_kps, info = centercircle_correct(kps, **kw)
    if not info["fit"] or (info["snapped_circle"] == 0 and info["snapped_middle"] == 0):
        return saved_params, info
    params = _solve(FramebyFrameCalib, new_kps)
    return (params if params is not None else saved_params), info


def goalframe_resolve(FramebyFrameCalib, kps, saved_params, policy="suppress",
                      group=GOAL_ALL, gate_thresh=2.0):
    """Return (params, info). info has n_goal, influential, delta, action."""
    n_goal = len([k for k in kps if k in group])
    info = {"n_goal": n_goal, "action": "baseline", "delta": 0.0, "influential": False}
    if n_goal == 0 or policy == "keep":
        return saved_params, info
    supp = _solve(FramebyFrameCalib, copy.deepcopy(kps), drop=group)
    if supp is None:
        return saved_params, info
    if policy == "suppress":
        info["action"] = "suppress"
        return supp, info
    # gated
    delta = _pose_delta(saved_params, supp)
    info["delta"] = delta
    if delta >= gate_thresh:
        info["influential"] = True
        info["action"] = "suppress_gated"
        return supp, info
    info["action"] = "baseline_gated"
    return saved_params, info
