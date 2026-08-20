"""Weighted camera solving for primitive-aware calibration.

Stage B (this file): **weighted candidate scoring** — reuse nbjw's solver to
generate the same candidate cameras it always considers (modes × ransac), but
select the winner by a **weighted reprojection score** over the keypoints (using
primitive reliability weights) instead of unweighted RMS. Zero risk: never moves
or deletes points; only changes which existing candidate is chosen.

Stage A (TODO): true weighted DLT / weighted PnP refine (custom solver).
"""
from __future__ import annotations
import copy
import numpy as np
try:
    import cv2
except Exception:
    cv2 = None

from . import primitive_mapping as PM


def ransac_inliers(kps, world, thresh=30.0):
    """NBJW-style outlier removal: RANSAC a ground-plane homography on the z=0
    keypoints, return the inlier id set (+ elevated goal pts, untouched). Kills
    gross/mirror hallucinations (e.g. left-goal point in a right-side view) that
    soft weighting can't reject. ponytail: fixed px thresh, tune if it over-prunes."""
    if cv2 is None:
        return set(kps)
    ids = [i for i in kps if i in world and i not in PM.ELEVATED_IDS]
    if len(ids) < 5:
        return set(kps)
    obj = np.array([world[i][:2] for i in ids], np.float32)
    img = np.array([[kps[i]['x'], kps[i]['y']] for i in ids], np.float32)
    H, mask = cv2.findHomography(obj, img, cv2.RANSAC, thresh)
    if mask is None or mask.sum() < 4:
        return set(kps)
    inl = {ids[k] for k in range(len(ids)) if mask[k]}
    inl |= {i for i in kps if i in PM.ELEVATED_IDS}
    return inl

_MODES = ['full', 'ground_plane', 'main']
_RANSACS = [0, 5, 10, 15, 25, 50]


def H_from_params(p, ptr_fn):
    pan = np.deg2rad(p['pan_degrees']); tilt = np.deg2rad(p['tilt_degrees']); roll = np.deg2rad(p['roll_degrees'])
    R = np.transpose(ptr_fn(pan, tilt, roll))
    pos = np.array(p['position_meters'], float)
    It = np.eye(4)[:-1]; It[:, -1] = -pos
    Q = np.array([[p['x_focal_length'], 0, p['principal_point'][0]],
                  [0, p['y_focal_length'], p['principal_point'][1]], [0, 0, 1]])
    P = Q @ (R @ It)
    H = P[:, [0, 1, 3]]
    if abs(H[-1, -1]) < 1e-12:
        return None
    return H / H[-1, -1]


def collect_candidates(FbF, kps, ptr_fn):
    """Run nbjw get_cam_params over all modes×ransac; return list of dicts."""
    cands = []
    cam = FbF(1920, 1080, denormalize=False)
    cam.update({k: {'x': v['x'], 'y': v['y']} for k, v in kps.items()})
    for mode in _MODES:
        for r in _RANSACS:
            try:
                cp, ret = cam.get_cam_params(mode=mode, use_ransac=r)
            except Exception:
                cp, ret = None, None
            if cp and ret:
                cands.append({"mode": mode, "ransac": r, "rep_err": float(ret), "cam_params": cp})
    return cands


def weighted_reproj_score(cam_params, kps, weights, world, ptr_fn):
    """Weighted mean reprojection error (px) of GROUND keypoints under candidate.
    Lower = better. Returns (score, coverage_weight)."""
    H = H_from_params(cam_params, ptr_fn)
    if H is None:
        return np.inf, 0.0
    num = den = 0.0
    for i, v in kps.items():
        if i not in world or i in PM.ELEVATED_IDS:
            continue
        xw, yw, _ = world[i]
        q = H @ np.array([xw, yw, 1.0])
        if abs(q[2]) < 1e-9:
            continue
        pred = q[:2] / q[2]
        err = float(np.hypot(pred[0] - v['x'], pred[1] - v['y']))
        w = float(weights.get(i, 1.0))
        num += w * err
        den += w
    if den <= 0:
        return np.inf, 0.0
    return num / den, den


def select_weighted(cands, kps, weights, world, ptr_fn, coverage_lambda=0.0):
    """Pick candidate minimizing weighted reproj score (optionally - coverage bonus)."""
    best, best_score = None, np.inf
    for c in cands:
        s, cov = weighted_reproj_score(c["cam_params"], kps, weights, world, ptr_fn)
        score = s - coverage_lambda * cov
        c["_wscore"] = s; c["_cov"] = cov
        if score < best_score:
            best_score, best = score, c
    return best


def _project_3d(x, X, ptr_fn):
    """x=[pan,tilt,roll,fx,posx,posy,posz] (deg/px/m), fy=fx, pp=(960,540)."""
    R = np.transpose(ptr_fn(np.deg2rad(x[0]), np.deg2rad(x[1]), np.deg2rad(x[2])))
    pt = np.array(X) - np.array([x[4], x[5], x[6]])
    rp = R @ pt
    if rp[2] <= 1e-3:
        return None
    rp = rp / rp[2]
    return np.array([rp[0] * x[3] + 960.0, rp[1] * x[3] + 540.0])


def weighted_refine(init_params, kps, weights, world, ptr_fn,
                    f_scale=5.0, max_nfev=60, use_weights=True):
    """Weighted PnP nonlinear refine from nbjw init. Soft per-point weights replace
    RANSAC hard point-dropping; Huber robust loss. Does NOT move/delete keypoints."""
    from scipy.optimize import least_squares
    ids = [i for i in kps if i in world]
    if len(ids) < 4 or not init_params:
        return init_params

    def res(x):
        out = []
        for i in ids:
            q = _project_3d(x, world[i], ptr_fn)
            w = np.sqrt(max(weights.get(i, 1e-3), 1e-3)) if (use_weights and weights) else 1.0
            if q is None:
                out += [1e3, 1e3]
            else:
                out += [w * (q[0] - kps[i]['x']), w * (q[1] - kps[i]['y'])]
        return out

    x0 = [init_params['pan_degrees'], init_params['tilt_degrees'], init_params['roll_degrees'],
          init_params['x_focal_length'], *init_params['position_meters']]
    try:
        sol = least_squares(res, x0, loss='huber', f_scale=f_scale, max_nfev=max_nfev)
        x = sol.x
    except Exception:
        return init_params
    return {'pan_degrees': float(x[0]), 'tilt_degrees': float(x[1]), 'roll_degrees': float(x[2]),
            'x_focal_length': float(x[3]), 'y_focal_length': float(x[3]),
            'principal_point': [960.0, 540.0], 'position_meters': [float(x[4]), float(x[5]), float(x[6])],
            'radial_distortion': [0.] * 6, 'tangential_distortion': [0., 0.], 'thin_prism_distortion': [0.] * 4}


def solve_variants(FbF, kps, saved_params, weights, world, ptr_fn):
    """Return dict of camera params for: raw(saved), unweighted-pick, weighted-pick.
    Also which candidate index won, for diagnostics."""
    cands = collect_candidates(FbF, kps, ptr_fn)
    out = {"n_cands": len(cands), "raw": saved_params}
    if not cands:
        out["unweighted"] = saved_params
        out["weighted"] = saved_params
        out["changed"] = False
        return out
    # unweighted pick = min unweighted reproj (all w=1)
    unit_w = {i: 1.0 for i in kps}
    uw = select_weighted([copy.copy(c) for c in cands], kps, unit_w, world, ptr_fn)
    ww = select_weighted([copy.copy(c) for c in cands], kps, weights, world, ptr_fn)
    out["unweighted"] = uw["cam_params"] if uw else saved_params
    out["weighted"] = ww["cam_params"] if ww else saved_params
    out["changed"] = (uw is not None and ww is not None and
                      (uw["mode"], uw["ransac"]) != (ww["mode"], ww["ransac"]))
    out["uw_pick"] = (uw["mode"], uw["ransac"]) if uw else None
    out["ww_pick"] = (ww["mode"], ww["ransac"]) if ww else None
    return out
