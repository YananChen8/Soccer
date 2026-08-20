"""Primitive-aware keypoint reliability weighting (创新点1 真实版).

不删点、不移点：只根据每个关键点与其所属 line / primitive / field 的几何一致性，
计算 reliability weight w_i，供 weighted candidate scoring / weighted solve 使用。

w_i = conf_i^a · c_line_i^b · c_primitive_i^c · c_field_i^d

关键设计（针对之前"自重投影残差全帧都低"的教训）：
  - line 一致性用 **leave-one-out 鲁棒拟合**：评 i 点时用同线其余点拟合直线，
    使真正的离群点暴露大残差（而非全局自洽掩盖）。
  - center circle 用 Sampson 距离到拟合椭圆，**不 snapping**。
  - box 角点用"所属两条边的交点"做 incidence 残差。
  - field 用同方向线族的 vanishing point 一致性。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from . import primitive_mapping as PM


@dataclass
class WeightConfig:
    sigma_line: float = 8.0       # px
    sigma_prim: float = 12.0      # px
    sigma_field: float = 0.05     # normalized VP-consistency scale
    a_conf: float = 1.0
    b_line: float = 1.0
    c_prim: float = 1.0
    d_field: float = 0.5
    min_line_pts: int = 3         # need >=this on a line to do leave-one-out
    use_conf: bool = True
    multi_line_reduce: str = "mean"   # how to combine a point's multiple line residuals: mean|min|max


# ── line fitting ──────────────────────────────────────────────────────────
def _fit_line(pts):
    """TLS line ax+by+c=0, |(a,b)|=1. pts: Nx2."""
    P = np.asarray(pts, float)
    c = P.mean(0)
    u, s, vt = np.linalg.svd(P - c)
    n = vt[-1]                      # normal = smallest singular direction
    a, b = n
    cc = -(a * c[0] + b * c[1])
    return np.array([a, b, cc])


def _pt_line_dist(p, line):
    return abs(line[0] * p[0] + line[1] * p[1] + line[2])


def _line_intersection(l1, l2):
    A = np.array([[l1[0], l1[1]], [l2[0], l2[1]]])
    bb = -np.array([l1[2], l2[2]])
    det = np.linalg.det(A)
    if abs(det) < 1e-9:
        return None
    return np.linalg.solve(A, bb)


# ── per-point line consistency (leave-one-out, robust) ─────────────────────
def straight_lines(world, tol=0.5):
    """Semantic lines whose 3D template points are actually collinear (excludes the
    Circle arcs, which must be handled by the ellipse, not line fitting)."""
    out = set()
    for line, ids in PM.KP_TO_LINE.items():
        P = [world[i][:2] for i in ids if i in world]
        if len(P) < 2:
            continue
        lf = _fit_line(P)
        if max(_pt_line_dist(p, lf) for p in P) <= tol:
            out.add(line)
    return out


def line_consistency(kps_img, cfg: WeightConfig, lines_to_fit):
    """kps_img: {id:(x,y)}. Returns {id: c_line in (0,1]} and per-line fit info.
    Only fits genuinely-straight lines (lines_to_fit)."""
    c_line = {}
    residuals = {}   # id -> list of (line, dist)
    line_info = {}
    for line in lines_to_fit:
        ids = PM.KP_TO_LINE[line]
        present = [i for i in ids if i in kps_img]
        if len(present) < cfg.min_line_pts:
            continue
        pts = {i: np.array(kps_img[i], float) for i in present}
        # robust full fit for line_info (coverage/MAD)
        full = _fit_line([pts[i] for i in present])
        dists_full = np.array([_pt_line_dist(pts[i], full) for i in present])
        line_info[line] = {"n": len(present), "median_resid": float(np.median(dists_full)),
                           "mad": float(np.median(np.abs(dists_full - np.median(dists_full))) * 1.4826)}
        # leave-one-out residual per point
        for i in present:
            others = [pts[j] for j in present if j != i]
            loo = _fit_line(others)
            d = _pt_line_dist(pts[i], loo)
            residuals.setdefault(i, []).append((line, d))
    for i, lst in residuals.items():
        ds = np.array([d for _l, d in lst])
        if cfg.multi_line_reduce == "min":
            d = ds.min()
        elif cfg.multi_line_reduce == "max":
            d = ds.max()
        else:
            d = ds.mean()
        c_line[i] = float(np.exp(-d * d / (2 * cfg.sigma_line ** 2)))
    return c_line, line_info, residuals


# ── center-circle Sampson consistency (no snapping) ────────────────────────
def _ellipse_sampson(pts, params):
    """Approx geometric distance of pts to ellipse (cx,cy,MA,ma,angle) via dense sampling."""
    (cx, cy), (MA, ma), ang = params
    a, b = MA / 2.0, ma / 2.0
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    ex = cx + a * np.cos(th) * ca - b * np.sin(th) * sa
    ey = cy + a * np.cos(th) * sa + b * np.sin(th) * ca
    E = np.stack([ex, ey], 1)
    out = []
    for p in pts:
        out.append(float(np.min(np.linalg.norm(E - p, axis=1))))
    return out


def circle_consistency(kps_img, cfg: WeightConfig):
    c = {}
    groups = {}
    if cv2 is None:
        return c, None
    # Circle central is the center circle; Circle left/right are the two penalty
    # arcs. Fitting them as one ellipse makes correct arc points look like
    # outliers whenever more than one arc is visible.
    for line in ["Circle central", "Circle left", "Circle right"]:
        ids = [i for i in PM.KP_TO_LINE[line] if i in kps_img]
        if len(ids) < 5:
            continue
        P = np.array([kps_img[i] for i in ids], np.float32)
        try:
            params = cv2.fitEllipse(P)
        except Exception:
            continue
        ds = _ellipse_sampson([np.array(kps_img[i], float) for i in ids], params)
        for i, d in zip(ids, ds):
            c[i] = float(np.exp(-d * d / (2 * cfg.sigma_prim ** 2)))
        groups[line] = {"ellipse": [list(params[0]), list(params[1]), float(params[2])], "n": len(ids)}
    return c, {"groups": groups} if groups else None


# ── box corner incidence: corner point vs intersection of its two edge lines ─
def goal_line_incidence(kps_img, line_fits, cfg: WeightConfig):
    """Goal-frame-specific check: do the goal-post BOTTOM points sit on the goal
    line (Side line left/right)? c in (0,1]. (Elevated post-tops can't be checked
    without the camera, so left at 1.)"""
    c = {}
    goalmap = {"left_goal_frame": ("Side line left", {13, 17}),
               "right_goal_frame": ("Side line right", {14, 18})}
    for prim, (gl, bottoms) in goalmap.items():
        if gl not in line_fits:
            continue
        for i in bottoms:
            if i in kps_img:
                d = _pt_line_dist(np.array(kps_img[i], float), line_fits[gl])
                c[i] = float(np.exp(-d * d / (2 * cfg.sigma_prim ** 2)))
    return c


def box_corner_consistency(kps_img, line_fits, cfg: WeightConfig):
    """For points on >=2 lines of the same box primitive, residual to lines' intersection."""
    c = {}
    for prim in ["left_penalty_box", "right_penalty_box", "left_goal_area", "right_goal_area"]:
        plines = [l for l in PM.PRIM_TO_LINES[prim] if l in line_fits]
        for i in PM.PRIM_TO_KPS[prim]:
            if i not in kps_img:
                continue
            mylines = [l for l in PM.KP_TO_LINES.get(i, []) if l in plines]
            if len(mylines) < 2:
                continue
            inter = _line_intersection(line_fits[mylines[0]], line_fits[mylines[1]])
            if inter is None:
                continue
            d = float(np.linalg.norm(np.array(kps_img[i], float) - inter))
            c[i] = float(np.exp(-d * d / (2 * cfg.sigma_prim ** 2)))
    return c


# ── field-level vanishing consistency of same-orientation line families ─────
def field_consistency(kps_img, world, cfg: WeightConfig):
    """Lines of the same world-orientation should share a vanishing point.
    Returns a single c_field per primitive (applied to its points)."""
    fits = {}
    for line, ids in PM.KP_TO_LINE.items():
        present = [i for i in ids if i in kps_img]
        if len(present) >= 2:
            fits[line] = _fit_line([np.array(kps_img[i], float) for i in present])
    fam = {"x": [], "y": []}
    for line, lf in fits.items():
        o = PM.line_orientation(line, {k: (v[0], v[1]) for k, v in world.items()})
        if o in fam:
            fam[o].append((line, lf))
    # vanishing point per family = least-squares intersection of its lines
    cfield_line = {}
    for o, group in fam.items():
        if len(group) < 2:
            continue
        Amat = np.array([lf[:2] for _l, lf in group])
        bvec = -np.array([lf[2] for _l, lf in group])
        vp, *_ = np.linalg.lstsq(Amat, bvec, rcond=None)
        # consistency of each line with the VP: normalized perpendicular distance
        for line, lf in group:
            d = abs(lf[0] * vp[0] + lf[1] * vp[1] + lf[2])
            scale = np.linalg.norm(vp) + 1e3
            cfield_line[line] = float(np.exp(-(d / scale) ** 2 / (2 * cfg.sigma_field ** 2)))
    # aggregate to primitive then to points
    c = {}
    for i in kps_img:
        lines = [l for l in PM.KP_TO_LINES.get(i, []) if l in cfield_line]
        if lines:
            c[i] = float(np.mean([cfield_line[l] for l in lines]))
    return c, cfield_line


# ── combine ────────────────────────────────────────────────────────────────
def compute_weights(kps_full, world, cfg: WeightConfig = None):
    """kps_full: {id:{'x','y','p'}} (saved pixel keypoints). Returns (weights{id:w}, debug)."""
    cfg = cfg or WeightConfig()
    kps_img = {i: (v['x'], v['y']) for i, v in kps_full.items()}
    sl = straight_lines(world)
    c_line, line_info, resid = line_consistency(kps_img, cfg, sl)
    line_fits = {l: _fit_line([np.array(kps_img[i], float) for i in PM.KP_TO_LINE[l] if i in kps_img])
                 for l in sl if sum(i in kps_img for i in PM.KP_TO_LINE[l]) >= 2}
    c_circ, circ_info = circle_consistency(kps_img, cfg)
    c_box = box_corner_consistency(kps_img, line_fits, cfg)
    c_goal = goal_line_incidence(kps_img, line_fits, cfg)
    c_field, _cf = field_consistency(kps_img, world, cfg)

    weights, dbg, comps = {}, {}, {}
    for i, v in kps_full.items():
        components = {
            "conf": float(v.get('p', 1.0)),
            "line": c_line.get(i, 1.0),
            "circle": c_circ.get(i, 1.0),
            "box": c_box.get(i, 1.0),
            "goal": c_goal.get(i, 1.0),
            "field": c_field.get(i, 1.0),
        }
        comps[i] = components
        cp_terms = [x for x in (c_circ.get(i), c_box.get(i)) if x is not None]
        cp = float(np.prod(cp_terms)) if cp_terms else 1.0
        conf = components["conf"] if cfg.use_conf else 1.0
        w = (conf ** cfg.a_conf) * (components["line"] ** cfg.b_line) * (cp ** cfg.c_prim) * (components["field"] ** cfg.d_field)
        weights[i] = float(w)
        dbg[i] = {**components, "w": float(w)}
    return weights, {"per_point": dbg, "components": comps, "line_info": line_info, "circle": circ_info}


def compose_weights(components_per_point, terms):
    """Build weights from a chosen subset of consistency terms (for ablations).
    terms: list from {conf,line,circle,box,goal,field}. w_i = product of chosen terms."""
    out = {}
    for i, comp in components_per_point.items():
        w = 1.0
        for t in terms:
            w *= float(comp.get(t, 1.0))
        out[i] = max(w, 1e-4)
    return out


def compose_weights_masked(components_per_point, terms, allowed_ids=None, base_for_other=0.0):
    """Variant weights where primitive-specific variants only optimize their own
    member points. Non-members get base_for_other instead of silently becoming
    confidence-only weights."""
    allowed = set(allowed_ids) if allowed_ids is not None else None
    out = {}
    for i, comp in components_per_point.items():
        if allowed is not None and i not in allowed:
            out[i] = max(float(base_for_other), 1e-4)
            continue
        w = 1.0
        for t in terms:
            w *= float(comp.get(t, 1.0))
        out[i] = max(w, 1e-4)
    return out
