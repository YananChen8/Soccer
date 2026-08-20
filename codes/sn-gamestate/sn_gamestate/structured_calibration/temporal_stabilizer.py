"""Structure-aware Temporal Pose Stabilizer (innovation #2).

Operates on the per-frame camera `parameters` dict (pan/tilt/roll/focal/position).
The SoccerNet calibration metric rebuilds the camera rotation purely from
pan/tilt/roll (it ignores any stored rotation_matrix), so smoothing these
interpretable scalars directly drives both the official line-accuracy metric and
the projection/BEV stability -- exactly the state representation the design doc
recommends ("don't average the 9 H elements").

Pipeline per video (frames processed in temporal order):
    raw_t                 -- raw params from the base calibrator (may be missing)
    pred_t = 2*s_{t-1} - s_{t-2}        (constant-velocity prediction)
    q_t                   -- frame confidence in [0,1] from structure + consistency
    alpha_t = base_alpha * q_t
    s_t (final) = alpha_t * raw_t + (1-alpha_t) * pred_t
Continuous low confidence (>= reinit_patience frames) -> reinitialize on raw.

The confidence q_t for the v1 (no Field Graph yet) combines:
  * structural proxy  s_struct: monotone in #predicted keypoints
        (calib_relation.json: corr(n_kp, line_err) ~ -0.71)
  * consistency proxy c_cons : how close raw_t is to the temporal prediction,
        normalized per-component by that video's robust scale (MAD of raw diffs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Order matters: this is the smoothed state vector layout.
STATE_FIELDS = [
    "pan_degrees", "tilt_degrees", "roll_degrees",
    "x_focal_length", "y_focal_length",
    "pos_x", "pos_y", "pos_z",
]
N_STATE = len(STATE_FIELDS)
ANGLE_IDX = [0, 1, 2]  # pan/tilt/roll are degrees (pan needs wrap handling)


@dataclass
class StabilizerConfig:
    # Selective gating: trust raw almost fully on confident frames (preserve
    # accuracy), blend toward the temporal prediction only on low-confidence /
    # jumpy frames. alpha is mapped from q via a ramp:
    #   q >= alpha_q_hi -> alpha = base_alpha   (near-raw, tiny smoothing)
    #   q <= alpha_q_lo -> alpha = min_alpha    (mostly prediction)
    base_alpha: float = 0.95          # weight on raw for fully-confident frames
    min_alpha: float = 0.10           # weight on raw for least-confident frames
    alpha_q_hi: float = 0.55          # q at/above which alpha = base_alpha
    alpha_q_lo: float = 0.15          # q at/below which alpha = min_alpha
    n_kp_lo: float = 4.0              # #kp at/below which structural conf -> 0
    n_kp_hi: float = 10.0             # #kp at/above which structural conf -> 1
    consistency_k: float = 3.0        # jump (in robust sigmas) where c_cons ~ 0.6
    q_low: float = 0.30               # below this q_t counts as "low confidence"
    reinit_patience: int = 5          # consecutive low-conf frames -> reinitialize
    enabled: bool = True


def _params_to_vec(p: dict) -> Optional[np.ndarray]:
    if not isinstance(p, dict) or len(p) == 0:
        return None
    try:
        pos = p.get("position_meters", [np.nan, np.nan, np.nan])
        v = np.array([
            float(p["pan_degrees"]), float(p["tilt_degrees"]), float(p["roll_degrees"]),
            float(p["x_focal_length"]), float(p["y_focal_length"]),
            float(pos[0]), float(pos[1]), float(pos[2]),
        ], dtype=float)
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if not np.all(np.isfinite(v)):
        return None
    return v


def _vec_to_params(v: np.ndarray, template: dict) -> dict:
    """Build a valid params dict, copying principal_point / distortions from template."""
    out = dict(template)  # shallow copy keeps principal_point, distortions, etc.
    out["pan_degrees"] = float(v[0])
    out["tilt_degrees"] = float(v[1])
    out["roll_degrees"] = float(v[2])
    out["x_focal_length"] = float(v[3])
    out["y_focal_length"] = float(v[4])
    out["position_meters"] = [float(v[5]), float(v[6]), float(v[7])]
    # rotation_matrix in the dict is ignored by from_json_parameters, but drop it
    # to avoid confusing any consumer that *does* read it.
    out.pop("rotation_matrix", None)
    return out


def _wrap_deg(d: np.ndarray) -> np.ndarray:
    """Wrap angle differences into (-180, 180] so pan near +/-180 interpolates sanely."""
    return (d + 180.0) % 360.0 - 180.0


def _robust_scales(raw_vecs: list[Optional[np.ndarray]]) -> np.ndarray:
    """Per-component robust scale = MAD of frame-to-frame raw diffs (with floors)."""
    arr = np.array([v for v in raw_vecs if v is not None], dtype=float)
    if arr.shape[0] < 2:
        return np.array([2.0, 2.0, 1.0, 100.0, 100.0, 0.5, 0.5, 0.5])
    diffs = np.diff(arr, axis=0)
    diffs[:, ANGLE_IDX] = _wrap_deg(diffs[:, ANGLE_IDX])
    mad = np.median(np.abs(diffs), axis=0) * 1.4826
    floors = np.array([0.5, 0.5, 0.3, 30.0, 30.0, 0.2, 0.2, 0.2])
    return np.maximum(mad, floors)


class TemporalStabilizer:
    """Stateful, per-video. Call reset() between videos, then step() per frame."""

    def __init__(self, cfg: StabilizerConfig, scales: np.ndarray):
        self.cfg = cfg
        self.scales = scales
        self.reset()

    def reset(self):
        self.prev1: Optional[np.ndarray] = None
        self.prev2: Optional[np.ndarray] = None
        self.low_streak = 0

    def _struct_conf(self, n_kp: int) -> float:
        lo, hi = self.cfg.n_kp_lo, self.cfg.n_kp_hi
        if hi <= lo:
            return 1.0
        return float(np.clip((n_kp - lo) / (hi - lo), 0.0, 1.0))

    def _consistency_conf(self, raw: np.ndarray, pred: Optional[np.ndarray]) -> float:
        if pred is None or raw is None:
            return 1.0
        d = raw - pred
        d[ANGLE_IDX] = _wrap_deg(d[ANGLE_IDX])
        jump = float(np.sqrt(np.mean((d / self.scales) ** 2)))
        return float(np.exp(-0.5 * (jump / self.cfg.consistency_k) ** 2))

    def step(self, raw: Optional[np.ndarray], n_kp: int, q_struct_ext: Optional[float] = None):
        """Return (final_vec or None, info dict) for one frame.
        q_struct_ext: optional external structural confidence in [0,1] (e.g. from
        self-reprojection consistency); when given it overrides the n_kp proxy."""
        cfg = self.cfg
        # prediction from accepted history
        if self.prev1 is not None and self.prev2 is not None:
            pred = 2.0 * self.prev1 - self.prev2
            pred[ANGLE_IDX] = self.prev1[ANGLE_IDX] + _wrap_deg(self.prev1[ANGLE_IDX] - self.prev2[ANGLE_IDX])
        elif self.prev1 is not None:
            pred = self.prev1.copy()
        else:
            pred = None

        info = {"alpha": None, "q": None, "s_struct": None, "c_cons": None,
                "state": "init", "reinit": False, "had_raw": raw is not None}

        # No usable raw -> coast on prediction (or nothing if no history)
        if raw is None:
            if pred is None:
                return None, info
            final = pred
            self.low_streak += 1
            info.update(alpha=0.0, q=0.0, s_struct=0.0, c_cons=0.0, state="fallback_norawn")
            self._push(final)
            return final, info

        s_struct = self._struct_conf(n_kp) if q_struct_ext is None else float(np.clip(q_struct_ext, 0.0, 1.0))
        c_cons = self._consistency_conf(raw, pred)
        q = s_struct * c_cons
        info.update(q=q, s_struct=s_struct, c_cons=c_cons)

        low = q < cfg.q_low
        self.low_streak = self.low_streak + 1 if low else 0

        # Reinitialize after sustained low confidence: trust raw, drop velocity term.
        if self.low_streak >= cfg.reinit_patience:
            final = raw.copy()
            self.prev2 = None
            self.prev1 = final
            self.low_streak = 0
            info.update(alpha=1.0, state="reinit", reinit=True)
            return final, info

        if pred is None:
            final = raw.copy()
            info.update(alpha=1.0, state="bootstrap")
            self._push(final)
            return final, info

        # selective ramp: confident -> base_alpha (near raw); unconfident -> min_alpha
        t = (q - cfg.alpha_q_lo) / max(cfg.alpha_q_hi - cfg.alpha_q_lo, 1e-6)
        t = float(np.clip(t, 0.0, 1.0))
        alpha = cfg.min_alpha + (cfg.base_alpha - cfg.min_alpha) * t
        final = alpha * raw + (1.0 - alpha) * pred
        final[ANGLE_IDX] = pred[ANGLE_IDX] + alpha * _wrap_deg(raw[ANGLE_IDX] - pred[ANGLE_IDX])
        info.update(alpha=alpha, state=("fallback" if low else "normal"))
        self._push(final)
        return final, info

    def _push(self, final: np.ndarray):
        self.prev2 = self.prev1
        self.prev1 = final.copy()


def stabilize_video(rows, n_kp_list, cfg: StabilizerConfig, quality_list=None):
    """rows: list of (frame, image_id, params_dict) sorted by frame.
    n_kp_list: parallel list of #keypoints per row (int).
    quality_list: optional parallel list of external structural conf in [0,1].
    Returns list of (frame, image_id, stabilized_params_dict_or_None, info)."""
    raw_vecs = [_params_to_vec(r[2]) for r in rows]
    scales = _robust_scales(raw_vecs)
    stab = TemporalStabilizer(cfg, scales)
    if quality_list is None:
        quality_list = [None] * len(rows)
    out = []
    for (frame, image_id, params), rawv, nkp, qext in zip(rows, raw_vecs, n_kp_list, quality_list):
        final_vec, info = stab.step(rawv, nkp, q_struct_ext=qext)
        if final_vec is None:
            out.append((frame, image_id, None, info))
        else:
            template = params if isinstance(params, dict) and len(params) else _last_valid_template(rows)
            out.append((frame, image_id, _vec_to_params(final_vec, template), info))
    return out


def hampel_stabilize_video(rows, window=7, n_sigma=3.0, fill_missing=True):
    """Robust spike-removal temporal filter (offline, non-causal).

    For each parameter component, a frame is replaced by its local median ONLY
    when it deviates from that median by > n_sigma * MAD (a Hampel identifier).
    Genuine sustained motion (smooth ramps) is left untouched; isolated bad-frame
    spikes are removed. Missing frames are linearly interpolated (improves
    completeness without being flagged as outliers).

    Returns list of (frame, image_id, params_or_None, info)."""
    n = len(rows)
    raw = np.full((n, N_STATE), np.nan)
    for i, (_f, _i, p) in enumerate(rows):
        v = _params_to_vec(p)
        if v is not None:
            raw[i] = v

    # work in unwrapped angle space so pan near +/-180 doesn't create fake spikes
    work = raw.copy()
    for j in ANGLE_IDX:
        col = work[:, j]
        mask = ~np.isnan(col)
        if mask.sum() >= 2:
            col[mask] = np.degrees(np.unwrap(np.radians(col[mask])))
            work[:, j] = col

    # linear-interpolate missing values (per component) for filtering
    idx = np.arange(n)
    filled = work.copy()
    for j in range(N_STATE):
        col = filled[:, j]
        m = ~np.isnan(col)
        if m.sum() == 0:
            continue
        if m.sum() < n:
            col[~m] = np.interp(idx[~m], idx[m], col[m])
            filled[:, j] = col

    out_arr = filled.copy()
    replaced = np.zeros(n, dtype=bool)
    w = window
    for i in range(n):
        lo, hi = max(0, i - w), min(n, i + w + 1)
        for j in range(N_STATE):
            seg = filled[lo:hi, j]
            med = np.median(seg)
            mad = np.median(np.abs(seg - med)) * 1.4826
            if mad > 1e-9 and abs(filled[i, j] - med) > n_sigma * mad:
                out_arr[i, j] = med
                replaced[i] = True

    # re-wrap angles back into (-180,180]
    for j in ANGLE_IDX:
        out_arr[:, j] = _wrap_deg(out_arr[:, j])

    out = []
    for i, (frame, image_id, params) in enumerate(rows):
        had_raw = _params_to_vec(params) is not None
        if not had_raw and not fill_missing:
            out.append((frame, image_id, None, {"state": "missing", "replaced": False}))
            continue
        template = params if (isinstance(params, dict) and len(params)) else _last_valid_template(rows)
        out.append((frame, image_id, _vec_to_params(out_arr[i], template),
                    {"state": "spike_removed" if replaced[i] else ("filled" if not had_raw else "kept"),
                     "replaced": bool(replaced[i]), "had_raw": had_raw}))
    return out


def _last_valid_template(rows):
    for _, _, p in rows:
        if isinstance(p, dict) and len(p):
            return p
    return {"principal_point": [960.0, 540.0],
            "radial_distortion": [0, 0, 0, 0, 0, 0],
            "tangential_distortion": [0, 0],
            "thin_prism_distortion": [0, 0, 0, 0]}
