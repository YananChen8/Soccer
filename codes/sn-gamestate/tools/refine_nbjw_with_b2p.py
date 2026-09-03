#!/usr/bin/env python3
"""Refine NBJW homographies with Broadcast2Pitch geometry.

This tool keeps SoccerMaster detections, tracks, ReID, role, team, and jersey
fields fixed. It only updates image-level homographies and recomputes
``bbox_pitch`` from the final image-to-pitch homography.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WIDTH, HEIGHT = 1920, 1080
TEMPLATE_CENTER = np.array([62.5, 39.0], dtype=float)
B2P_TEMPLATE_TO_SOCCER = np.array(
    [
        [1.0, 0.0, -62.5],
        [0.0, 1.0, -39.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)
SOCCER_TO_B2P_TEMPLATE = np.linalg.inv(B2P_TEMPLATE_TO_SOCCER)
CENTRAL_CIRCLE_RADIUS = 9.14
PITCH_X_LIMIT = 52.5
PITCH_Y_LIMIT = 34.0

LINE_NAMES = [
    "Big rect. left bottom",
    "Big rect. left main",
    "Big rect. left top",
    "Big rect. right bottom",
    "Big rect. right main",
    "Big rect. right top",
    "Circle central",
    "Middle line",
    "Side line bottom",
    "Side line left",
    "Side line right",
    "Side line top",
    "Small rect. left bottom",
    "Small rect. left main",
    "Small rect. left top",
    "Small rect. right bottom",
    "Small rect. right main",
    "Small rect. right top",
    "Circle left",
    "Circle right",
    "All lines",
]

PITCH_LINE_ENDPOINTS_TEMPLATE = {
    "Big rect. left top": [[10.0, 18.84, 1.0], [26.50, 18.84, 1.0]],
    "Big rect. left bottom": [[10.0, 59.16, 1.0], [26.50, 59.16, 1.0]],
    "Big rect. left main": [[26.50, 59.16, 1.0], [26.50, 18.84, 1.0]],
    "Big rect. right top": [[98.50, 18.84, 1.0], [115.0, 18.84, 1.0]],
    "Big rect. right bottom": [[98.50, 59.16, 1.0], [115.0, 59.16, 1.0]],
    "Big rect. right main": [[98.50, 59.16, 1.0], [98.50, 18.84, 1.0]],
    "Small rect. left top": [[10.0, 29.84, 1.0], [15.5, 29.84, 1.0]],
    "Small rect. left bottom": [[10.0, 48.16, 1.0], [15.5, 48.16, 1.0]],
    "Small rect. left main": [[15.5, 48.16, 1.0], [15.5, 29.84, 1.0]],
    "Small rect. right top": [[109.50, 29.84, 1.0], [115.0, 29.84, 1.0]],
    "Small rect. right bottom": [[109.50, 48.16, 1.0], [115.0, 48.16, 1.0]],
    "Small rect. right main": [[109.50, 48.16, 1.0], [109.50, 29.84, 1.0]],
    "Side line bottom": [[10.0, 73.0, 1.0], [115.0, 73.0, 1.0]],
    "Side line top": [[10.0, 5.0, 1.0], [115.0, 5.0, 1.0]],
    "Side line left": [[10.0, 73.0, 1.0], [10.0, 5.0, 1.0]],
    "Side line right": [[115.0, 73.0, 1.0], [115.0, 5.0, 1.0]],
    "Middle line": [[62.50, 73.0, 1.0], [62.50, 5.0, 1.0]],
}

PITCH_LINE_KEYPOINTS = {
    "Big rect. left bottom": [6, 19],
    "Big rect. left top": [1, 15],
    "Big rect. left main": [15, 16, 17, 18, 19],
    "Big rect. right bottom": [81, 95],
    "Big rect. right top": [77, 90],
    "Big rect. right main": [77, 78, 79, 80, 81],
    "Small rect. left bottom": [5, 11],
    "Small rect. left top": [2, 9],
    "Small rect. left main": [9, 10, 11],
    "Small rect. right bottom": [87, 94],
    "Small rect. right top": [85, 91],
    "Small rect. right main": [85, 86, 87],
    "Side line bottom": [7, 12, 20, 28, 35, 42, 51, 60, 67, 74, 82, 88, 96],
    "Side line top": [0, 8, 14, 22, 29, 36, 45, 54, 61, 68, 76, 84, 89],
    "Side line left": [0, 1, 2, 3, 4, 5, 6, 7],
    "Side line right": [89, 90, 91, 92, 93, 94, 95, 96],
    "Middle line": [45, 46, 47, 48, 49, 50, 51],
    "Circle central": [39, 57],
    "Circle left": [16, 18, 21],
    "Circle right": [75, 78, 80],
}

IMAGE_DIAGNOSTIC_COLUMNS = [
    "h_nbjw",
    "h_refined",
    "b2p_num_keypoints",
    "b2p_num_lines",
    "b2p_circle_available",
    "b2p_raw_residual",
    "b2p_refined_residual",
    "b2p_homography_delta",
    "b2p_solver_success",
    "b2p_accepted",
    "b2p_fallback_reason",
]

REQUIRED_METRIC_COLUMNS = [
    "video",
    "frame",
    "num_keypoints",
    "num_lines",
    "circle_available",
    "raw_residual",
    "refined_residual",
    "homography_delta",
    "solver_success",
    "accepted",
    "fallback_reason",
]


class StateDiagnosisError(RuntimeError):
    """Raised when the source state cannot be refined safely."""


@dataclass(frozen=True)
class StateFrame:
    member_id: str
    image_index: Any
    image_id: Any
    video: str
    frame: int
    image_path: Path


@dataclass
class FrameObservation:
    keypoints: np.ndarray
    line_points: dict[str, np.ndarray]
    line_confidences: dict[str, float]
    circle_points: np.ndarray
    cache_path: Path
    error: Optional[str] = None


@dataclass(frozen=True)
class RefinementConfig:
    wl: float = 1.0
    wc: float = 1.0
    wk: float = 1.0
    trust_weight: float = 1.0
    trust_pixel_scale: float = 75.0
    min_keypoints: int = 4
    min_lines: int = 2
    min_line_points: int = 8
    min_circle_points: int = 8
    keypoint_conf: float = 0.4
    line_conf: float = 0.8
    max_line_points_per_class: int = 40
    max_lm_nfev: int = 200
    improve_eps: float = 1e-6
    max_anchor_shift_px: float = 120.0
    max_pitch_abs_x: float = 80.0
    max_pitch_abs_y: float = 60.0
    min_pitch_range_fraction: float = 0.75


@dataclass
class FrameRefinementResult:
    final_h: Optional[np.ndarray]
    refined_h: Optional[np.ndarray]
    raw_residual: Optional[float]
    refined_residual: Optional[float]
    homography_delta: Optional[float]
    solver_success: bool
    accepted: bool
    fallback_reason: str
    num_keypoints: int
    num_lines: int
    circle_available: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--b2p-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--videos", nargs="*", default=[])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-state", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--b2p-python", type=Path, default=None)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--b2p-device", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--wl", type=float, default=1.0)
    parser.add_argument("--wc", type=float, default=1.0)
    parser.add_argument("--wk", type=float, default=1.0)
    parser.add_argument("--trust-weight", type=float, default=1.0)
    parser.add_argument("--trust-pixel-scale", type=float, default=75.0)
    parser.add_argument("--min-keypoints", type=int, default=4)
    parser.add_argument("--min-lines", type=int, default=2)
    parser.add_argument("--min-line-points", type=int, default=8)
    parser.add_argument("--min-circle-points", type=int, default=8)
    parser.add_argument("--keypoint-conf", type=float, default=0.4)
    parser.add_argument("--line-conf", type=float, default=0.8)
    parser.add_argument("--max-line-points-per-class", type=int, default=40)
    parser.add_argument("--max-lm-nfev", type=int, default=200)
    parser.add_argument("--max-anchor-shift-px", type=float, default=120.0)
    parser.add_argument("--max-pitch-abs-x", type=float, default=80.0)
    parser.add_argument("--max-pitch-abs-y", type=float, default=60.0)
    parser.add_argument("--overlay-dir", type=Path, default=None)
    parser.add_argument("--num-overlays", type=int, default=20)
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--eval-out", type=Path, default=None)
    parser.add_argument("--eval-threshold", type=float, default=5.0)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument(
        "--infer-cache-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def normalize_video(value: Any) -> str:
    text = str(scalar(value)).strip()
    match = re.search(r"SNGS[-_]?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"SNGS-{int(match.group(1)):03d}"
    if text.isdigit():
        return f"SNGS-{int(text):03d}"
    return text


def video_suffix(video: str) -> str:
    match = re.search(r"(\d+)$", normalize_video(video))
    return f"{int(match.group(1)):03d}" if match else str(video)


def scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if type(value).__name__ == "NAType":
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


def parse_frame(value: Any) -> Optional[int]:
    if is_missing(value):
        return None
    value = scalar(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(value)
    text = Path(str(value)).stem
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    token = nums[-1]
    if len(token) > 6:
        token = token[-6:]
    return int(token)


def dataframe_rows(df: pd.DataFrame) -> Iterable[tuple[Any, dict[str, Any]]]:
    for index, row in df.iterrows():
        record = row.to_dict()
        if "id" not in record:
            record["id"] = index
        yield index, record


def safe_npz_stem(video: str, frame: int) -> str:
    return f"{normalize_video(video).replace('-', '_')}_{int(frame):06d}.npz"


def cache_path_for_frame(cache_dir: Path, video: str, frame: int) -> Path:
    return Path(cache_dir) / normalize_video(video) / safe_npz_stem(video, frame)


def split_root(dataset_root: Path, split: str) -> Path:
    root = Path(dataset_root)
    return root / split if (root / split).exists() else root


def infer_video_from_record(record: Mapping[str, Any], member_id: str) -> Optional[str]:
    for key in ("video", "video_name", "name", "file_path"):
        value = record.get(key)
        if not is_missing(value):
            text = str(scalar(value))
            match = re.search(r"SNGS[-_]?(\d+)", text, flags=re.IGNORECASE)
            if match:
                return f"SNGS-{int(match.group(1)):03d}"
    if re.fullmatch(r"\d+", str(member_id)):
        return normalize_video(member_id)
    if re.search(r"SNGS[-_]?(\d+)", str(member_id), flags=re.IGNORECASE):
        return normalize_video(member_id)
    return None


def resolve_image_path(
    raw_path: Any,
    dataset_root: Path,
    split: str,
    video: str,
    frame: int,
) -> Path:
    if not is_missing(raw_path):
        path = Path(str(raw_path))
        if path.exists():
            return path
        stem = path.name
    else:
        stem = f"{frame:06d}.jpg"
    root = split_root(dataset_root, split)
    img_dir = root / normalize_video(video) / "img1"
    candidates = [
        img_dir / stem,
        img_dir / f"{frame:06d}.jpg",
        img_dir / f"{frame:06d}.png",
        img_dir / f"{frame}.jpg",
        img_dir / f"{frame}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def member_kind(member: str) -> tuple[Optional[str], Optional[str]]:
    if not member.endswith(".pkl"):
        return None, None
    stem = member[:-4]
    if stem.endswith("_image"):
        return "image", stem[: -len("_image")]
    return "detection", stem


def read_pickle(zf: zipfile.ZipFile, member: str) -> Any:
    with zf.open(member, "r") as fh:
        return pickle.load(fh)


def collect_state_frames(
    state_pklz: Path,
    dataset_root: Path,
    split: str,
    videos: Sequence[str],
    max_frames: Optional[int] = None,
) -> list[StateFrame]:
    wanted = {normalize_video(v) for v in videos} if videos else None
    frames: list[StateFrame] = []
    with zipfile.ZipFile(state_pklz, "r") as zf:
        for member in sorted(zf.namelist()):
            kind, member_id = member_kind(member)
            if kind != "image" or member_id is None:
                continue
            image_df = read_pickle(zf, member)
            if not isinstance(image_df, pd.DataFrame):
                continue
            for index, record in dataframe_rows(image_df):
                video = infer_video_from_record(record, member_id)
                if video is None and wanted and len(wanted) == 1:
                    video = next(iter(wanted))
                if video is None:
                    continue
                video = normalize_video(video)
                if wanted and video not in wanted:
                    continue
                frame = first_not_none(
                    parse_frame(record.get("file_path")),
                    parse_frame(record.get("frame")),
                    parse_frame(record.get("id")),
                    parse_frame(record.get("image_id")),
                    parse_frame(index),
                )
                if frame is None:
                    continue
                image_id = record.get("id", index)
                image_path = resolve_image_path(
                    record.get("file_path"),
                    dataset_root,
                    split,
                    video,
                    frame,
                )
                frames.append(
                    StateFrame(
                        member_id=str(member_id),
                        image_index=index,
                        image_id=image_id,
                        video=video,
                        frame=int(frame),
                        image_path=image_path,
                    )
                )
    frames.sort(key=lambda item: (item.video, item.frame, str(item.image_id)))
    if max_frames is not None:
        frames = frames[: max(0, max_frames)]
    return frames


def first_not_none(*values: Optional[Any]) -> Optional[Any]:
    for value in values:
        if value is not None:
            return value
    return None


def normalize_homography(h: Any) -> Optional[np.ndarray]:
    if h is None or is_missing(h):
        return None
    try:
        arr = np.asarray(h, dtype=float)
    except Exception:
        return None
    if arr.shape != (3, 3) or not np.all(np.isfinite(arr)):
        return None
    if abs(float(arr[2, 2])) < 1e-12:
        return None
    return arr / float(arr[2, 2])


def homography_to_params(h: np.ndarray) -> np.ndarray:
    hn = normalize_homography(h)
    if hn is None:
        raise ValueError("Invalid homography.")
    return hn.reshape(-1)[:8].astype(float)


def params_to_homography(params: Sequence[float]) -> np.ndarray:
    p = np.asarray(params, dtype=float)
    if p.shape != (8,):
        raise ValueError("Homography parameter vector must have shape (8,).")
    return np.array(
        [
            [p[0], p[1], p[2]],
            [p[3], p[4], p[5]],
            [p[6], p[7], 1.0],
        ],
        dtype=float,
    )


def invert_homography(h: np.ndarray) -> Optional[np.ndarray]:
    hn = normalize_homography(h)
    if hn is None:
        return None
    try:
        inv = np.linalg.inv(hn)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(inv)):
        return None
    return inv / inv[2, 2] if abs(inv[2, 2]) > 1e-12 else inv


def project_points(h: np.ndarray, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=float)
    pts_h = np.concatenate([pts[:, :2], ones], axis=1)
    proj = (np.asarray(h, dtype=float) @ pts_h.T).T
    denom = proj[:, 2:3]
    finite = np.isfinite(proj).all(axis=1) & (np.abs(denom[:, 0]) > 1e-12)
    out = np.full((pts.shape[0], 2), np.nan, dtype=float)
    out[finite] = proj[finite, :2] / denom[finite]
    return out, finite


def transform_template_points_to_soccer(points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    xy, finite = project_points(B2P_TEMPLATE_TO_SOCCER, pts[:, :2])
    if not np.all(finite):
        raise ValueError("Template to SoccerMaster conversion produced invalid points.")
    return xy


def convert_b2p_homography_to_soccer(
    h_b2p: np.ndarray,
    direction: str,
) -> np.ndarray:
    """Convert a B2P homography to image -> SoccerMaster pitch coordinates.

    ``direction`` must be explicit. For image -> template, use C @ H_b2p.
    For template -> image, invert first and then apply C.
    """
    hn = normalize_homography(h_b2p)
    if hn is None:
        raise ValueError("Invalid B2P homography.")
    if direction == "image_to_template":
        return normalize_homography(B2P_TEMPLATE_TO_SOCCER @ hn)
    if direction == "template_to_image":
        inv = invert_homography(hn)
        if inv is None:
            raise ValueError("Invalid template-to-image B2P homography.")
        return normalize_homography(B2P_TEMPLATE_TO_SOCCER @ inv)
    raise ValueError("direction must be 'image_to_template' or 'template_to_image'.")


def verify_b2p_conversion_with_synthetic_points(h_b2p: np.ndarray, direction: str) -> bool:
    template_pts = np.array(
        [
            [10.0, 5.0],
            [115.0, 5.0],
            [115.0, 73.0],
            [10.0, 73.0],
        ],
        dtype=float,
    )
    if direction == "image_to_template":
        h_img_to_template = normalize_homography(h_b2p)
    elif direction == "template_to_image":
        h_img_to_template = invert_homography(h_b2p)
    else:
        return False
    if h_img_to_template is None:
        return False
    h_template_to_img = invert_homography(h_img_to_template)
    h_img_to_soccer = convert_b2p_homography_to_soccer(h_b2p, direction)
    if h_template_to_img is None or h_img_to_soccer is None:
        return False
    image_pts, ok_img = project_points(h_template_to_img, template_pts)
    soccer_expected = transform_template_points_to_soccer(template_pts)
    soccer_actual, ok_soccer = project_points(h_img_to_soccer, image_pts)
    back_image, ok_back = project_points(invert_homography(h_img_to_soccer), soccer_expected)
    return bool(
        np.all(ok_img)
        and np.all(ok_soccer)
        and np.all(ok_back)
        and np.allclose(soccer_actual, soccer_expected, atol=1e-7)
        and np.allclose(back_image, image_pts, atol=1e-7)
    )


def line_from_points(points_xy: np.ndarray) -> np.ndarray:
    p = np.asarray(points_xy, dtype=float)
    p0 = np.array([p[0, 0], p[0, 1], 1.0], dtype=float)
    p1 = np.array([p[1, 0], p[1, 1], 1.0], dtype=float)
    line = np.cross(p0, p1)
    norm = np.linalg.norm(line[:2])
    if norm <= 1e-12:
        raise ValueError("Degenerate line.")
    return line / norm


def line_model_soccer(line_name: str) -> Optional[np.ndarray]:
    endpoints = PITCH_LINE_ENDPOINTS_TEMPLATE.get(line_name)
    if endpoints is None:
        return None
    pts = transform_template_points_to_soccer(np.asarray(endpoints, dtype=float)[:, :2])
    return line_from_points(pts)


def template_keypoints_soccer(template_kpts: np.ndarray) -> np.ndarray:
    kpts = np.asarray(template_kpts, dtype=float)
    if kpts.ndim != 2 or kpts.shape[1] < 2:
        raise ValueError("Template keypoints must have shape (N, >=2).")
    return transform_template_points_to_soccer(kpts[:, :2])


def standard_anchor_points_soccer() -> np.ndarray:
    template_pts = np.array(
        [
            [10.0, 5.0],
            [62.5, 5.0],
            [115.0, 5.0],
            [10.0, 39.0],
            [62.5, 39.0],
            [115.0, 39.0],
            [10.0, 73.0],
            [62.5, 73.0],
            [115.0, 73.0],
            [21.0, 39.0],
            [104.0, 39.0],
            [26.5, 18.84],
            [98.5, 59.16],
        ],
        dtype=float,
    )
    return transform_template_points_to_soccer(template_pts)


ANCHOR_POINTS_SOCCER = standard_anchor_points_soccer()


def valid_keypoint_mask(keypoints: np.ndarray, conf_threshold: float) -> np.ndarray:
    if keypoints.size == 0:
        return np.zeros((0,), dtype=bool)
    pts = np.asarray(keypoints, dtype=float)
    present = (np.abs(pts[:, 0]) > 1e-9) | (np.abs(pts[:, 1]) > 1e-9)
    return present & (pts[:, 2] >= conf_threshold)


def sample_rows_evenly(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    idx = np.linspace(0, len(points) - 1, limit).round().astype(int)
    return points[idx]


def count_observations(obs: FrameObservation, cfg: RefinementConfig) -> tuple[int, int, bool]:
    kpt_count = int(valid_keypoint_mask(obs.keypoints, cfg.keypoint_conf).sum())
    line_count = 0
    for name, pts in obs.line_points.items():
        if name not in PITCH_LINE_ENDPOINTS_TEMPLATE:
            continue
        conf = obs.line_confidences.get(name, 0.0)
        if conf >= cfg.line_conf and len(pts) >= cfg.min_line_points:
            line_count += 1
    circle_available = len(obs.circle_points) >= cfg.min_circle_points
    return kpt_count, line_count, bool(circle_available)


def observation_image_points(obs: FrameObservation, cfg: RefinementConfig) -> np.ndarray:
    pieces: list[np.ndarray] = []
    mask = valid_keypoint_mask(obs.keypoints, cfg.keypoint_conf)
    if len(mask) and mask.any():
        pieces.append(obs.keypoints[mask, :2])
    for name, pts in obs.line_points.items():
        if name not in PITCH_LINE_ENDPOINTS_TEMPLATE:
            continue
        conf = obs.line_confidences.get(name, 0.0)
        if conf >= cfg.line_conf and len(pts) >= cfg.min_line_points:
            pieces.append(sample_rows_evenly(np.asarray(pts, dtype=float), cfg.max_line_points_per_class)[:, :2])
    if len(obs.circle_points) >= cfg.min_circle_points:
        pieces.append(np.asarray(obs.circle_points, dtype=float)[:, :2])
    if not pieces:
        return np.zeros((0, 2), dtype=float)
    return np.concatenate(pieces, axis=0)


def geometric_residuals_from_h(
    h: np.ndarray,
    obs: FrameObservation,
    template_kpts_soccer: np.ndarray,
    cfg: RefinementConfig,
) -> np.ndarray:
    res: list[float] = []

    for line_name, pts in obs.line_points.items():
        line = line_model_soccer(line_name)
        if line is None:
            continue
        line_conf = obs.line_confidences.get(line_name, 0.0)
        if line_conf < cfg.line_conf or len(pts) < cfg.min_line_points:
            continue
        line_pts = sample_rows_evenly(np.asarray(pts, dtype=float), cfg.max_line_points_per_class)
        projected, finite = project_points(h, line_pts[:, :2])
        point_weights = np.sqrt(np.maximum(line_pts[:, 2], 0.0) * max(line_conf, 1e-6) * cfg.wl)
        for xy, ok, weight in zip(projected, finite, point_weights):
            if not ok:
                res.append(1e3)
                continue
            res.append(float(weight * (line[0] * xy[0] + line[1] * xy[1] + line[2])))

    circle = np.asarray(obs.circle_points, dtype=float)
    if len(circle) >= cfg.min_circle_points:
        projected, finite = project_points(h, circle[:, :2])
        weights = np.sqrt(np.maximum(circle[:, 2], 0.0) * cfg.wc)
        for xy, ok, weight in zip(projected, finite, weights):
            if not ok:
                res.append(1e3)
                continue
            res.append(float(weight * (np.linalg.norm(xy) - CENTRAL_CIRCLE_RADIUS)))

    keypoints = np.asarray(obs.keypoints, dtype=float)
    if keypoints.size:
        mask = valid_keypoint_mask(keypoints, cfg.keypoint_conf)
        valid_indices = np.nonzero(mask)[0]
        if len(valid_indices):
            valid_indices = valid_indices[valid_indices < len(template_kpts_soccer)]
        if len(valid_indices):
            projected, finite = project_points(h, keypoints[valid_indices, :2])
            expected = template_kpts_soccer[valid_indices, :2]
            weights = np.sqrt(np.maximum(keypoints[valid_indices, 2], 0.0) * cfg.wk)
            for xy, target, ok, weight in zip(projected, expected, finite, weights):
                if not ok:
                    res.extend([1e3, 1e3])
                    continue
                delta = weight * (xy - target)
                res.extend([float(delta[0]), float(delta[1])])

    return np.asarray(res, dtype=float)


def rms_residual(residuals: np.ndarray) -> Optional[float]:
    if residuals.size == 0:
        return None
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        return None
    return float(np.sqrt(np.mean(finite * finite)))


def trust_residuals_from_h(h: np.ndarray, h0: np.ndarray, cfg: RefinementConfig) -> np.ndarray:
    inv_h = invert_homography(h)
    inv_h0 = invert_homography(h0)
    if inv_h is None or inv_h0 is None:
        return np.full((ANCHOR_POINTS_SOCCER.shape[0] * 2,), 1e3, dtype=float)
    new_img, ok_new = project_points(inv_h, ANCHOR_POINTS_SOCCER)
    old_img, ok_old = project_points(inv_h0, ANCHOR_POINTS_SOCCER)
    if not (np.all(ok_new) and np.all(ok_old)):
        return np.full((ANCHOR_POINTS_SOCCER.shape[0] * 2,), 1e3, dtype=float)
    scale = max(cfg.trust_pixel_scale, 1e-6)
    return (math.sqrt(max(cfg.trust_weight, 0.0)) * (new_img - old_img) / scale).reshape(-1)


def anchor_shift_px(h: np.ndarray, h0: np.ndarray) -> Optional[float]:
    inv_h = invert_homography(h)
    inv_h0 = invert_homography(h0)
    if inv_h is None or inv_h0 is None:
        return None
    new_img, ok_new = project_points(inv_h, ANCHOR_POINTS_SOCCER)
    old_img, ok_old = project_points(inv_h0, ANCHOR_POINTS_SOCCER)
    if not (np.all(ok_new) and np.all(ok_old)):
        return None
    return float(np.max(np.linalg.norm(new_img - old_img, axis=1)))


def projection_range_ok(h: np.ndarray, obs: FrameObservation, cfg: RefinementConfig) -> bool:
    pts = observation_image_points(obs, cfg)
    if len(pts) == 0:
        return False
    projected, finite = project_points(h, pts)
    if not finite.any():
        return False
    projected = projected[finite]
    in_range = (
        (np.abs(projected[:, 0]) <= cfg.max_pitch_abs_x)
        & (np.abs(projected[:, 1]) <= cfg.max_pitch_abs_y)
    )
    return float(in_range.mean()) >= cfg.min_pitch_range_fraction


def default_least_squares() -> Callable[..., Any]:
    try:
        from scipy.optimize import least_squares
    except Exception as exc:  # pragma: no cover - exercised in target env only
        raise RuntimeError(
            "scipy is required for LM refinement. Run the refinement in the "
            "SoccerMaster environment or preinstall scipy there."
        ) from exc
    return least_squares


def refine_homography_with_observations(
    h0_raw: Any,
    obs: FrameObservation,
    template_kpts_soccer: np.ndarray,
    cfg: RefinementConfig,
    least_squares_fn: Optional[Callable[..., Any]] = None,
) -> FrameRefinementResult:
    h0 = normalize_homography(h0_raw)
    kpt_count, line_count, circle_available = count_observations(obs, cfg)
    if h0 is None:
        return FrameRefinementResult(
            final_h=None,
            refined_h=None,
            raw_residual=None,
            refined_residual=None,
            homography_delta=None,
            solver_success=False,
            accepted=False,
            fallback_reason="missing_or_invalid_h0",
            num_keypoints=kpt_count,
            num_lines=line_count,
            circle_available=circle_available,
        )
    if obs.error:
        return fallback_result(h0, kpt_count, line_count, circle_available, f"b2p_cache_error:{obs.error}")
    if kpt_count < cfg.min_keypoints:
        return fallback_result(h0, kpt_count, line_count, circle_available, "too_few_keypoints")
    if line_count < cfg.min_lines:
        return fallback_result(h0, kpt_count, line_count, circle_available, "too_few_lines")

    raw_vec = geometric_residuals_from_h(h0, obs, template_kpts_soccer, cfg)
    raw_residual = rms_residual(raw_vec)
    if raw_vec.size < 8 or raw_residual is None:
        return fallback_result(h0, kpt_count, line_count, circle_available, "too_few_residuals")

    least_squares_impl = least_squares_fn or default_least_squares()

    def residual_fn(params: np.ndarray) -> np.ndarray:
        h = params_to_homography(params)
        geom = geometric_residuals_from_h(h, obs, template_kpts_soccer, cfg)
        trust = trust_residuals_from_h(h, h0, cfg)
        return np.concatenate([geom, trust])

    try:
        result = least_squares_impl(
            residual_fn,
            homography_to_params(h0),
            method="lm",
            max_nfev=cfg.max_lm_nfev,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
    except Exception as exc:
        return fallback_result(
            h0,
            kpt_count,
            line_count,
            circle_available,
            f"solver_exception:{type(exc).__name__}",
            raw_residual=raw_residual,
        )

    solver_success = bool(getattr(result, "success", False))
    if not solver_success:
        return fallback_result(
            h0,
            kpt_count,
            line_count,
            circle_available,
            "solver_failed",
            raw_residual=raw_residual,
        )

    h_refined = normalize_homography(params_to_homography(np.asarray(result.x, dtype=float)))
    if h_refined is None or invert_homography(h_refined) is None:
        return fallback_result(
            h0,
            kpt_count,
            line_count,
            circle_available,
            "invalid_refined_h",
            raw_residual=raw_residual,
            refined_h=h_refined,
            solver_success=True,
        )

    refined_vec = geometric_residuals_from_h(h_refined, obs, template_kpts_soccer, cfg)
    refined_residual = rms_residual(refined_vec)
    if refined_residual is None:
        return fallback_result(
            h0,
            kpt_count,
            line_count,
            circle_available,
            "missing_refined_residual",
            raw_residual=raw_residual,
            refined_h=h_refined,
            solver_success=True,
        )
    if refined_residual > raw_residual - cfg.improve_eps:
        return fallback_result(
            h0,
            kpt_count,
            line_count,
            circle_available,
            "not_better_than_nbjw",
            raw_residual=raw_residual,
            refined_residual=refined_residual,
            refined_h=h_refined,
            solver_success=True,
        )

    delta = anchor_shift_px(h_refined, h0)
    if delta is None:
        return fallback_result(
            h0,
            kpt_count,
            line_count,
            circle_available,
            "invalid_anchor_projection",
            raw_residual=raw_residual,
            refined_residual=refined_residual,
            refined_h=h_refined,
            solver_success=True,
        )
    if delta > cfg.max_anchor_shift_px:
        return fallback_result(
            h0,
            kpt_count,
            line_count,
            circle_available,
            "anchor_shift_too_large",
            raw_residual=raw_residual,
            refined_residual=refined_residual,
            homography_delta=delta,
            refined_h=h_refined,
            solver_success=True,
        )
    if not projection_range_ok(h_refined, obs, cfg):
        return fallback_result(
            h0,
            kpt_count,
            line_count,
            circle_available,
            "projection_out_of_range",
            raw_residual=raw_residual,
            refined_residual=refined_residual,
            homography_delta=delta,
            refined_h=h_refined,
            solver_success=True,
        )

    return FrameRefinementResult(
        final_h=h_refined,
        refined_h=h_refined,
        raw_residual=raw_residual,
        refined_residual=refined_residual,
        homography_delta=delta,
        solver_success=True,
        accepted=True,
        fallback_reason="",
        num_keypoints=kpt_count,
        num_lines=line_count,
        circle_available=circle_available,
    )


def fallback_result(
    h0: np.ndarray,
    kpt_count: int,
    line_count: int,
    circle_available: bool,
    reason: str,
    raw_residual: Optional[float] = None,
    refined_residual: Optional[float] = None,
    homography_delta: Optional[float] = None,
    refined_h: Optional[np.ndarray] = None,
    solver_success: bool = False,
) -> FrameRefinementResult:
    return FrameRefinementResult(
        final_h=normalize_homography(h0),
        refined_h=refined_h,
        raw_residual=raw_residual,
        refined_residual=refined_residual,
        homography_delta=homography_delta,
        solver_success=solver_success,
        accepted=False,
        fallback_reason=reason,
        num_keypoints=kpt_count,
        num_lines=line_count,
        circle_available=circle_available,
    )


def slugify_line_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def empty_observation(cache_path: Path, error: Optional[str] = None) -> FrameObservation:
    return FrameObservation(
        keypoints=np.zeros((97, 3), dtype=np.float32),
        line_points={},
        line_confidences={name: 0.0 for name in LINE_NAMES},
        circle_points=np.zeros((0, 3), dtype=np.float32),
        cache_path=cache_path,
        error=error,
    )


def save_observation_npz(path: Path, obs: FrameObservation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "keypoints": np.asarray(obs.keypoints, dtype=np.float32),
        "circle_points": np.asarray(obs.circle_points, dtype=np.float32),
        "line_names": np.asarray(LINE_NAMES),
        "line_confidences": np.asarray([obs.line_confidences.get(name, 0.0) for name in LINE_NAMES], dtype=np.float32),
        "line_point_names": np.asarray(list(obs.line_points.keys())),
        "error": np.asarray(obs.error or ""),
    }
    for name, pts in obs.line_points.items():
        arrays[f"line_points__{slugify_line_name(name)}"] = np.asarray(pts, dtype=np.float32)
    np.savez_compressed(path, **arrays)


def load_observation_npz(path: Path) -> FrameObservation:
    if not path.exists():
        return empty_observation(path, error="missing_cache")
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        return empty_observation(path, error=f"bad_cache:{type(exc).__name__}")
    line_names = [str(x) for x in data.get("line_names", np.asarray(LINE_NAMES))]
    line_conf_array = np.asarray(data.get("line_confidences", np.zeros((len(line_names),))), dtype=float)
    line_confidences = {
        name: float(line_conf_array[i]) if i < len(line_conf_array) else 0.0
        for i, name in enumerate(line_names)
    }
    line_points: dict[str, np.ndarray] = {}
    for name in [str(x) for x in data.get("line_point_names", np.asarray([]))]:
        key = f"line_points__{slugify_line_name(name)}"
        if key in data:
            line_points[name] = np.asarray(data[key], dtype=np.float32)
    error = str(np.asarray(data.get("error", np.asarray(""))).item() or "")
    return FrameObservation(
        keypoints=np.asarray(data.get("keypoints", np.zeros((97, 3))), dtype=np.float32),
        line_points=line_points,
        line_confidences=line_confidences,
        circle_points=np.asarray(data.get("circle_points", np.zeros((0, 3))), dtype=np.float32),
        cache_path=path,
        error=error or None,
    )


def ensure_cache_for_frames(frames: Sequence[StateFrame], args: argparse.Namespace) -> None:
    missing = [
        frame
        for frame in frames
        if not cache_path_for_frame(args.cache_dir, frame.video, frame.frame).exists()
    ]
    if not missing:
        return
    if args.skip_inference:
        raise FileNotFoundError(
            f"{len(missing)} B2P cache files are missing and --skip-inference was set. "
            f"First missing: {cache_path_for_frame(args.cache_dir, missing[0].video, missing[0].frame)}"
        )
    if args.b2p_python is not None:
        command = [
            str(args.b2p_python),
            str(Path(__file__).resolve()),
            "--infer-cache-only",
            "--source-state",
            str(args.source_state),
            "--dataset-root",
            str(args.dataset_root),
            "--b2p-root",
            str(args.b2p_root),
            "--checkpoint",
            str(args.checkpoint),
            "--split",
            str(args.split),
            "--cache-dir",
            str(args.cache_dir),
            "--out-state",
            str(args.out_state),
            "--metrics-out",
            str(args.metrics_out),
        ]
        if args.videos:
            command.append("--videos")
            command.extend(str(v) for v in args.videos)
        if args.max_frames is not None:
            command.extend(["--max-frames", str(args.max_frames)])
        if args.b2p_device:
            command.extend(["--b2p-device", str(args.b2p_device)])
        subprocess.run(command, check=True)
        return
    run_b2p_inference_to_cache(missing, args)


def run_b2p_inference_to_cache(frames: Sequence[StateFrame], args: argparse.Namespace) -> None:
    try:
        import cv2
        import torch
        import torch.nn.functional as torch_f
    except Exception as exc:  # pragma: no cover - target env dependent
        raise RuntimeError(
            "B2P cache is missing and the current Python cannot import torch/cv2. "
            "Run once with --b2p-python pointing at the b2p_localization Python, "
            "or precompute the .npz cache in that environment."
        ) from exc

    sys.path.insert(0, str(args.b2p_root))
    try:
        import kpts as b2p_kpts
    except Exception as exc:  # pragma: no cover - target env dependent
        raise RuntimeError(f"Could not import Broadcast2Pitch kpts.py from {args.b2p_root}") from exc
    finally:
        try:
            sys.path.remove(str(args.b2p_root))
        except ValueError:
            pass

    device = args.b2p_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = b2p_kpts.Unet(out_ch=98, num_lines=21)
    state_dict = torch.load(str(args.checkpoint), map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    template_kpts = np.load(str(Path(args.b2p_root) / "template" / "soccernet_template_97.npy"))
    if template_kpts.shape[0] < 97:
        raise ValueError(f"Unexpected B2P template shape: {template_kpts.shape}")

    for state_frame in frames:
        out_path = cache_path_for_frame(args.cache_dir, state_frame.video, state_frame.frame)
        if out_path.exists():
            continue
        if not state_frame.image_path.exists():
            save_observation_npz(out_path, empty_observation(out_path, error="image_missing"))
            continue
        image = cv2.imread(str(state_frame.image_path))
        if image is None:
            save_observation_npz(out_path, empty_observation(out_path, error="image_read_failed"))
            continue
        obs = predict_single_b2p_observation(image, model, b2p_kpts, torch, torch_f, device, out_path, args)
        save_observation_npz(out_path, obs)


def predict_single_b2p_observation(
    image: np.ndarray,
    model: Any,
    b2p_kpts: Any,
    torch: Any,
    torch_f: Any,
    device: str,
    cache_path: Path,
    args: argparse.Namespace,
) -> FrameObservation:
    hm_size = (384, 384)
    img_height, img_width = image.shape[:2]
    img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    img_tensor = torch_f.interpolate(
        img_tensor.unsqueeze(0),
        size=hm_size,
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img_tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img_tensor.dtype).view(1, 3, 1, 1)
    img_tensor = ((img_tensor - mean) / std).to(device)

    with torch.no_grad():
        pred_kpts, pred_lines = model(img_tensor)
        preds_batch, maxvals_batch = b2p_kpts.get_final_preds_torch(pred_kpts)

    scale_x = img_width / float(hm_size[0])
    scale_y = img_height / float(hm_size[1])
    keypoints = []
    distance_threshold = 30.0
    for kp_idx in range(97):
        conf = float(maxvals_batch[0, kp_idx].item())
        if conf >= args.keypoint_conf:
            pred_kp = preds_batch[0, kp_idx]
            x = int(np.rint(pred_kp[0] * scale_x))
            y = int(np.rint(pred_kp[1] * scale_y))
            current = (x, y, conf)
            if keypoints:
                dists = [
                    np.linalg.norm(np.array(current[:2], dtype=float) - np.array(prev[:2], dtype=float))
                    for prev in keypoints
                ]
                keypoints.append(current if min(dists) > distance_threshold else (0, 0, 0.0))
            else:
                keypoints.append(current)
        else:
            keypoints.append((0, 0, 0.0))
    keypoints_arr = np.asarray(keypoints, dtype=np.float32)

    confidence_scores = torch.max(pred_lines.view(pred_lines.size(0), pred_lines.size(1), -1), dim=2)[0]
    line_confidences = {
        name: float(confidence_scores[0, idx].detach().cpu().item())
        for idx, name in enumerate(LINE_NAMES)
    }
    line_detected = {
        name: (line_confidences.get(name, 0.0) >= args.line_conf)
        for name in LINE_NAMES
        if name != "All lines"
    }
    apply_b2p_left_right_veto(line_detected)

    all_lines_idx = LINE_NAMES.index("All lines")
    all_lines_heatmap = pred_lines[0, all_lines_idx].detach().cpu().numpy()
    line_points: dict[str, np.ndarray] = {}
    circle_points = np.zeros((0, 3), dtype=np.float32)

    for line_name in LINE_NAMES:
        if line_name == "All lines" or not line_detected.get(line_name, False):
            continue
        line_idx = LINE_NAMES.index(line_name)
        heatmap = pred_lines[0, line_idx].detach().cpu().numpy()
        if line_name == "Circle central":
            points = b2p_kpts.extract_circle_points_from_heatmap(
                heatmap,
                threshold=0.9,
                high_intensity_threshold=0.96,
                num_samples=0,
                num_random_points=100,
            )
            if len(points):
                circle_points = scale_b2p_heatmap_points(points, scale_x, scale_y)
        elif line_name in PITCH_LINE_ENDPOINTS_TEMPLATE:
            points = b2p_kpts.extract_line_points_from_heatmap(
                heatmap,
                all_lines_heatmap=all_lines_heatmap,
                threshold=0.93,
                high_intensity_threshold=0.9,
                num_samples=100,
                num_random_points=100,
                min_length=15,
                all_lines_weight=0.3,
            )
            if len(points):
                line_points[line_name] = scale_b2p_heatmap_points(points, scale_x, scale_y)

    return FrameObservation(
        keypoints=keypoints_arr,
        line_points=line_points,
        line_confidences=line_confidences,
        circle_points=circle_points,
        cache_path=cache_path,
    )


def apply_b2p_left_right_veto(line_detected: dict[str, bool]) -> None:
    left_keys = {
        "Big rect. left bottom",
        "Big rect. left main",
        "Big rect. left top",
        "Small rect. left bottom",
        "Small rect. left main",
        "Small rect. left top",
    }
    right_keys = {
        "Big rect. right bottom",
        "Big rect. right main",
        "Big rect. right top",
        "Small rect. right bottom",
        "Small rect. right main",
        "Small rect. right top",
    }
    full_pitch = line_detected.get("Middle line", False) and line_detected.get("Circle central", False)
    left_count = sum(bool(line_detected.get(key, False)) for key in left_keys)
    right_count = sum(bool(line_detected.get(key, False)) for key in right_keys)
    if full_pitch:
        return
    if left_count >= 2 and left_count >= 2.0 * right_count:
        for key in right_keys:
            line_detected[key] = False
    elif right_count >= 2 and right_count >= 2.0 * left_count:
        for key in left_keys:
            line_detected[key] = False


def scale_b2p_heatmap_points(points: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    scaled = np.asarray(points, dtype=np.float32).copy()
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return scaled


def bbox_ltwh_to_pitch(h: Optional[np.ndarray], bbox_ltwh: Any) -> Optional[dict[str, float]]:
    hn = normalize_homography(h)
    if hn is None or is_missing(bbox_ltwh):
        return None
    try:
        l, t, w, hgt = [float(v) for v in np.asarray(bbox_ltwh, dtype=float).reshape(-1)[:4]]
    except Exception:
        return None
    if w <= 0 or hgt <= 0:
        return None
    points = np.array(
        [
            [l, t + hgt],
            [l + w, t + hgt],
            [l + w / 2.0, t + hgt],
        ],
        dtype=float,
    )
    projected, finite = project_points(hn, points)
    if not np.all(finite) or not np.all(np.isfinite(projected)):
        return None
    return {
        "x_bottom_left": float(projected[0, 0]),
        "y_bottom_left": float(projected[0, 1]),
        "x_bottom_right": float(projected[1, 0]),
        "y_bottom_right": float(projected[1, 1]),
        "x_bottom_middle": float(projected[2, 0]),
        "y_bottom_middle": float(projected[2, 1]),
    }


def update_bbox_pitch_for_member(
    det_df: pd.DataFrame,
    h_by_image_key: Mapping[str, Optional[np.ndarray]],
    update_image_keys: Optional[set[str]] = None,
) -> pd.DataFrame:
    if "bbox_ltwh" not in det_df.columns:
        return det_df
    out = det_df.copy()
    if "bbox_pitch" not in out.columns:
        out["bbox_pitch"] = None
    for index, row in out.iterrows():
        image_id = row.get("image_id")
        image_key = str(scalar(image_id))
        if update_image_keys is not None and image_key not in update_image_keys:
            continue
        h = h_by_image_key.get(image_key)
        if h is None:
            continue
        out.at[index, "bbox_pitch"] = bbox_ltwh_to_pitch(h, row.get("bbox_ltwh"))
    return out


def inspect_state_has_h(state_pklz: Path, videos: Sequence[str]) -> tuple[bool, str]:
    if not Path(state_pklz).exists():
        return False, f"source state not found: {state_pklz}"
    wanted = {normalize_video(v) for v in videos} if videos else None
    image_members = 0
    selected_members = 0
    missing_h_members: list[str] = []
    missing_parameters_members: list[str] = []
    valid_h_count = 0
    with zipfile.ZipFile(state_pklz, "r") as zf:
        for member in zf.namelist():
            kind, member_id = member_kind(member)
            if kind != "image" or member_id is None:
                continue
            image_members += 1
            image_df = read_pickle(zf, member)
            member_videos = {
                normalize_video(v)
                for _, rec in dataframe_rows(image_df)
                for v in [infer_video_from_record(rec, member_id)]
                if v is not None
            }
            selected = not wanted or bool(member_videos & wanted) or (
                len(wanted) == 1 and normalize_video(member_id) in wanted
            )
            if not selected:
                continue
            selected_members += 1
            if "h" not in image_df.columns:
                missing_h_members.append(member)
                continue
            if "parameters" not in image_df.columns:
                missing_parameters_members.append(member)
            valid_h_count += sum(normalize_homography(v) is not None for v in image_df["h"].tolist())
    if image_members == 0:
        return False, "state contains no *_image.pkl members"
    if selected_members == 0:
        return False, "no selected video image members were found"
    if missing_h_members:
        return False, "missing h column in: " + ", ".join(missing_h_members[:8])
    if valid_h_count == 0:
        return False, "selected image members contain h column but no valid homography values"
    param_msg = (
        ", parameters column present"
        if not missing_parameters_members
        else ", missing parameters column in: " + ", ".join(missing_parameters_members[:8])
    )
    return True, f"selected image members={selected_members}, valid h frames={valid_h_count}{param_msg}"


def infer_state_h_direction(image_df: pd.DataFrame, det_df: pd.DataFrame) -> tuple[str, Optional[float], Optional[float]]:
    if "h" not in image_df.columns or "bbox_pitch" not in det_df.columns or "bbox_ltwh" not in det_df.columns:
        return "unknown", None, None
    h_by_key = {str(scalar(row.get("id", index))): row.get("h") for index, row in image_df.iterrows()}
    direct_errors: list[float] = []
    inverse_errors: list[float] = []
    for _, row in det_df.head(250).iterrows():
        pitch = row.get("bbox_pitch")
        if not isinstance(pitch, Mapping) or "x_bottom_middle" not in pitch or "y_bottom_middle" not in pitch:
            continue
        h = normalize_homography(h_by_key.get(str(scalar(row.get("image_id")))))
        if h is None:
            continue
        inv = invert_homography(h)
        if inv is None:
            continue
        try:
            l, t, w, hh = [float(v) for v in np.asarray(row.get("bbox_ltwh"), dtype=float).reshape(-1)[:4]]
        except Exception:
            continue
        point = np.array([[l + w / 2.0, t + hh]], dtype=float)
        target = np.array([float(pitch["x_bottom_middle"]), float(pitch["y_bottom_middle"])], dtype=float)
        direct, ok_direct = project_points(h, point)
        inverse, ok_inverse = project_points(inv, point)
        if ok_direct[0]:
            direct_errors.append(float(np.linalg.norm(direct[0] - target)))
        if ok_inverse[0]:
            inverse_errors.append(float(np.linalg.norm(inverse[0] - target)))
    if len(direct_errors) < 4 or len(inverse_errors) < 4:
        return "unknown", None, None
    direct_med = float(np.median(direct_errors))
    inverse_med = float(np.median(inverse_errors))
    if direct_med <= inverse_med * 0.1 or direct_med + 1e-3 < inverse_med:
        return "image_to_pitch", direct_med, inverse_med
    if inverse_med <= direct_med * 0.1 or inverse_med + 1e-3 < direct_med:
        return "pitch_to_image", direct_med, inverse_med
    return "ambiguous", direct_med, inverse_med


def member_selected(image_df: pd.DataFrame, member_id: str, wanted: Optional[set[str]]) -> bool:
    if not wanted:
        return True
    if normalize_video(member_id) in wanted:
        return True
    for _, record in dataframe_rows(image_df):
        video = infer_video_from_record(record, member_id)
        if video is not None and normalize_video(video) in wanted:
            return True
    return len(wanted) == 1 and str(member_id) == video_suffix(next(iter(wanted))).lstrip("0")


def build_refinement_config(args: argparse.Namespace) -> RefinementConfig:
    return RefinementConfig(
        wl=args.wl,
        wc=args.wc,
        wk=args.wk,
        trust_weight=args.trust_weight,
        trust_pixel_scale=args.trust_pixel_scale,
        min_keypoints=args.min_keypoints,
        min_lines=args.min_lines,
        min_line_points=args.min_line_points,
        min_circle_points=args.min_circle_points,
        keypoint_conf=args.keypoint_conf,
        line_conf=args.line_conf,
        max_line_points_per_class=args.max_line_points_per_class,
        max_lm_nfev=args.max_lm_nfev,
        max_anchor_shift_px=args.max_anchor_shift_px,
        max_pitch_abs_x=args.max_pitch_abs_x,
        max_pitch_abs_y=args.max_pitch_abs_y,
    )


def load_template_keypoints(args: argparse.Namespace) -> np.ndarray:
    template_path = Path(args.b2p_root) / "template" / "soccernet_template_97.npy"
    if not template_path.exists():
        raise FileNotFoundError(f"B2P template keypoints not found: {template_path}")
    return np.load(str(template_path))


def set_object_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        df[column] = pd.Series([None] * len(df), index=df.index, dtype=object)


def refine_state(args: argparse.Namespace) -> dict[str, Any]:
    ok, diagnosis = inspect_state_has_h(args.source_state, args.videos)
    if not ok:
        raise StateDiagnosisError(f"L0 state is not refinement-ready: {diagnosis}")

    selected_frames = collect_state_frames(
        args.source_state,
        args.dataset_root,
        args.split,
        args.videos,
        args.max_frames,
    )
    if not selected_frames:
        raise StateDiagnosisError("No frames selected from the source state.")
    ensure_cache_for_frames(selected_frames, args)

    template_kpts_soccer = template_keypoints_soccer(load_template_keypoints(args))
    cfg = build_refinement_config(args)
    frames_by_member: dict[str, set[str]] = {}
    for frame in selected_frames:
        frames_by_member.setdefault(frame.member_id, set()).add(str(scalar(frame.image_id)))

    wanted = {normalize_video(v) for v in args.videos} if args.videos else None
    metrics_rows: list[dict[str, Any]] = []
    output_members: dict[str, bytes] = {}
    member_direction: dict[str, tuple[str, Optional[float], Optional[float]]] = {}

    with zipfile.ZipFile(args.source_state, "r") as zf:
        names = zf.namelist()
        for member in names:
            if member == "summary.json":
                continue
            kind, member_id = member_kind(member)
            if kind is None or member_id is None:
                output_members[member] = zf.read(member)
                continue
            image_member = f"{member_id}_image.pkl"
            det_member = f"{member_id}.pkl"
            if member != image_member:
                continue
            image_df = read_pickle(zf, image_member)
            if not member_selected(image_df, member_id, wanted):
                output_members[det_member] = zf.read(det_member) if det_member in names else b""
                output_members[image_member] = zf.read(image_member)
                continue
            if det_member not in names:
                raise StateDiagnosisError(f"Missing detection member for selected image member: {det_member}")
            det_df = read_pickle(zf, det_member)

            direction, direct_err, inverse_err = infer_state_h_direction(image_df, det_df)
            member_direction[member_id] = (direction, direct_err, inverse_err)
            image_df_out = image_df.copy()
            for column in IMAGE_DIAGNOSTIC_COLUMNS:
                set_object_column(image_df_out, column)

            h_by_image_key: dict[str, Optional[np.ndarray]] = {}
            update_image_keys: set[str] = set()
            for index, record in dataframe_rows(image_df):
                video = infer_video_from_record(record, member_id)
                if video is None and wanted and len(wanted) == 1:
                    video = next(iter(wanted))
                if video is None:
                    video = normalize_video(member_id)
                video = normalize_video(video)
                frame = first_not_none(
                    parse_frame(record.get("file_path")),
                    parse_frame(record.get("frame")),
                    parse_frame(record.get("id")),
                    parse_frame(record.get("image_id")),
                    parse_frame(index),
                )
                image_id = record.get("id", index)
                h_original = normalize_homography(record.get("h"))
                h0 = h_original
                if direction == "pitch_to_image" and h0 is not None:
                    h0 = invert_homography(h0)

                selected = (not wanted or video in wanted) and (
                    args.max_frames is None
                    or str(scalar(image_id)) in frames_by_member.get(str(member_id), set())
                )
                if not selected or frame is None:
                    h_by_image_key[str(scalar(image_id))] = h0
                    continue

                cache_path = cache_path_for_frame(args.cache_dir, video, int(frame))
                obs = load_observation_npz(cache_path)
                result = refine_homography_with_observations(h0, obs, template_kpts_soccer, cfg)
                final_h = result.final_h

                image_df_out.at[index, "h_nbjw"] = h_original.copy() if h_original is not None else None
                image_df_out.at[index, "h_refined"] = final_h.copy() if final_h is not None else None
                image_df_out.at[index, "h"] = final_h.copy() if final_h is not None else record.get("h")
                image_df_out.at[index, "b2p_num_keypoints"] = result.num_keypoints
                image_df_out.at[index, "b2p_num_lines"] = result.num_lines
                image_df_out.at[index, "b2p_circle_available"] = bool(result.circle_available)
                image_df_out.at[index, "b2p_raw_residual"] = result.raw_residual
                image_df_out.at[index, "b2p_refined_residual"] = result.refined_residual
                image_df_out.at[index, "b2p_homography_delta"] = result.homography_delta
                image_df_out.at[index, "b2p_solver_success"] = bool(result.solver_success)
                image_df_out.at[index, "b2p_accepted"] = bool(result.accepted)
                image_df_out.at[index, "b2p_fallback_reason"] = result.fallback_reason

                h_by_image_key[str(scalar(image_id))] = final_h
                update_image_keys.add(str(scalar(image_id)))
                metrics_rows.append(
                    {
                        "video": video,
                        "frame": int(frame),
                        "num_keypoints": result.num_keypoints,
                        "num_lines": result.num_lines,
                        "circle_available": bool(result.circle_available),
                        "raw_residual": result.raw_residual,
                        "refined_residual": result.refined_residual,
                        "homography_delta": result.homography_delta,
                        "solver_success": bool(result.solver_success),
                        "accepted": bool(result.accepted),
                        "fallback_reason": result.fallback_reason,
                    }
                )

            det_df_out = update_bbox_pitch_for_member(det_df, h_by_image_key, update_image_keys)
            output_members[det_member] = pickle_to_bytes(det_df_out)
            output_members[image_member] = pickle_to_bytes(image_df_out)

        for member in names:
            if member == "summary.json" or member in output_members:
                continue
            kind, member_id = member_kind(member)
            if kind == "detection":
                paired = f"{member_id}_image.pkl"
                if paired in output_members:
                    continue
            output_members[member] = zf.read(member)

        summary_bytes = update_summary_bytes(zf, args, output_members)

    args.out_state.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out_state, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf_out:
        zf_out.writestr("summary.json", summary_bytes)
        for member in sorted(output_members):
            if output_members[member]:
                zf_out.writestr(member, output_members[member])

    write_metrics_csv(args.metrics_out, metrics_rows)
    invariance = validate_state_invariance(args.source_state, args.out_state)
    maybe_generate_overlays(args, selected_frames, args.out_state)
    eval_summary = maybe_evaluate_homography_states(args)

    return {
        "source_state": str(args.source_state),
        "out_state": str(args.out_state),
        "metrics_out": str(args.metrics_out),
        "state_diagnosis": diagnosis,
        "frames": len(metrics_rows),
        "accepted": sum(1 for row in metrics_rows if row["accepted"]),
        "fallback": sum(1 for row in metrics_rows if not row["accepted"]),
        "state_h_direction": member_direction,
        "state_invariance": invariance,
        "eval": eval_summary,
    }


def pickle_to_bytes(value: Any) -> bytes:
    import io

    buf = io.BytesIO()
    pickle.dump(value, buf, protocol=pickle.DEFAULT_PROTOCOL)
    return buf.getvalue()


def update_summary_bytes(zf: zipfile.ZipFile, args: argparse.Namespace, output_members: Mapping[str, bytes]) -> bytes:
    payload: dict[str, Any]
    if "summary.json" in zf.namelist():
        try:
            payload = json.loads(zf.read("summary.json").decode("utf-8"))
        except Exception:
            payload = {}
    else:
        payload = {}
    columns = payload.setdefault("columns", {})
    if isinstance(columns, list):
        det_cols = columns
        img_cols: list[str] = []
        payload["columns"] = {"detection": det_cols, "image": img_cols}
        columns = payload["columns"]
    if isinstance(columns, dict):
        det_cols = columns.setdefault("detection", [])
        img_cols = columns.setdefault("image", [])
        if "bbox_pitch" not in det_cols:
            det_cols.append("bbox_pitch")
        for column in ["h", *IMAGE_DIAGNOSTIC_COLUMNS]:
            if column not in img_cols:
                img_cols.append(column)
    payload.setdefault("state_patches", []).append(
        {
            "name": "nbjw_b2p_localization_refinement",
            "source": "NBJW h + Broadcast2Pitch line/circle/keypoint LM with anchor trust-region.",
        }
    )
    payload["nbjw_b2p_refinement"] = {
        "source_state": str(args.source_state),
        "dataset_root": str(args.dataset_root),
        "b2p_root": str(args.b2p_root),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "videos": [normalize_video(v) for v in args.videos],
        "cache_dir": str(args.cache_dir),
        "metrics_out": str(args.metrics_out),
        "coord_transform": B2P_TEMPLATE_TO_SOCCER.tolist(),
        "b2p_template_center": TEMPLATE_CENTER.tolist(),
        "homography_direction": "image_to_soccer_pitch",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = pd.DataFrame(rows, columns=REQUIRED_METRIC_COLUMNS)
        writer.to_csv(fh, index=False)


def value_equal(a: Any, b: Any) -> bool:
    if is_missing(a) and is_missing(b):
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))
        except TypeError:
            return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(value_equal(a[key], b[key]) for key in a)
    return a == b


def assert_column_equal(left: pd.Series, right: pd.Series, label: str) -> None:
    if len(left) != len(right):
        raise AssertionError(f"{label}: length changed")
    for idx, (a, b) in enumerate(zip(left.tolist(), right.tolist())):
        if not value_equal(a, b):
            raise AssertionError(f"{label}: value changed at position {idx}")


def validate_state_invariance(source_state: Path, out_state: Path) -> dict[str, Any]:
    allowed_image_changed = {"h", *IMAGE_DIAGNOSTIC_COLUMNS}
    allowed_detection_changed = {"bbox_pitch"}
    checked_members = 0
    with zipfile.ZipFile(source_state, "r") as zf_src, zipfile.ZipFile(out_state, "r") as zf_out:
        src_names = {name for name in zf_src.namelist() if name.endswith(".pkl")}
        out_names = {name for name in zf_out.namelist() if name.endswith(".pkl")}
        if src_names != out_names:
            raise AssertionError("state member set changed")
        for member in sorted(src_names):
            kind, _ = member_kind(member)
            src_df = read_pickle(zf_src, member)
            out_df = read_pickle(zf_out, member)
            if len(src_df) != len(out_df):
                raise AssertionError(f"{member}: row count changed")
            if kind == "detection":
                for required in ("bbox_ltwh", "track_id", "role", "team", "jersey"):
                    if required in src_df.columns or required in out_df.columns:
                        if required not in src_df.columns or required not in out_df.columns:
                            raise AssertionError(f"{member}: required column {required} added/removed")
                        assert_column_equal(src_df[required], out_df[required], f"{member}:{required}")
                for column in src_df.columns:
                    if column not in allowed_detection_changed:
                        assert_column_equal(src_df[column], out_df[column], f"{member}:{column}")
                unexpected = set(out_df.columns) - set(src_df.columns)
                if unexpected:
                    raise AssertionError(f"{member}: unexpected detection columns {sorted(unexpected)}")
            elif kind == "image":
                unexpected = set(out_df.columns) - set(src_df.columns) - IMAGE_DIAGNOSTIC_COLUMNS_SET()
                if unexpected:
                    raise AssertionError(f"{member}: unexpected image columns {sorted(unexpected)}")
                for column in src_df.columns:
                    if column not in allowed_image_changed:
                        assert_column_equal(src_df[column], out_df[column], f"{member}:{column}")
            checked_members += 1
    return {"passed": True, "checked_members": checked_members}


def IMAGE_DIAGNOSTIC_COLUMNS_SET() -> set[str]:
    return set(IMAGE_DIAGNOSTIC_COLUMNS)


def load_gt_lines(dataset_root: Path, split: str, video: str) -> dict[str, Any]:
    labels = split_root(dataset_root, split) / normalize_video(video) / "Labels-GameState.json"
    if not labels.exists():
        return {}
    with labels.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    out: dict[str, Any] = {}
    for ann in data.get("annotations", []):
        if ann.get("supercategory") != "pitch":
            continue
        image_id = str(scalar(ann.get("image_id")))
        out[image_id] = ann.get("lines", {})
    return out


def maybe_generate_overlays(args: argparse.Namespace, frames: Sequence[StateFrame], out_state: Path) -> None:
    if args.no_overlays or args.num_overlays <= 0:
        return
    overlay_dir = args.overlay_dir
    if overlay_dir is None:
        overlay_dir = args.out_state.parent.parent / "overlays"
    selected = evenly_spaced_frames(frames, args.num_overlays)
    if not selected:
        return
    try:
        import cv2
    except Exception:
        print("Overlay generation skipped: cv2 is not available.", file=sys.stderr)
        return
    with zipfile.ZipFile(args.source_state, "r") as zf_src, zipfile.ZipFile(out_state, "r") as zf_out:
        source_h = load_h_by_video_frame(zf_src, selected)
        refined_h = load_h_by_video_frame(zf_out, selected)
    gt_by_video = {video: load_gt_lines(args.dataset_root, args.split, video) for video in sorted({f.video for f in selected})}
    overlay_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for frame in selected:
        if not frame.image_path.exists():
            continue
        image = cv2.imread(str(frame.image_path))
        if image is None:
            continue
        h0 = source_h.get((frame.video, frame.frame))
        h1 = refined_h.get((frame.video, frame.frame))
        gt_lines = gt_by_video.get(frame.video, {}).get(str(scalar(frame.image_id)), {})
        panel_nbjw = image.copy()
        panel_refined = image.copy()
        panel_gt = image.copy()
        draw_projected_pitch(cv2, panel_nbjw, h0, (0, 0, 255))
        draw_projected_pitch(cv2, panel_refined, h1, (0, 180, 0))
        draw_gt_lines(cv2, panel_gt, gt_lines, image.shape[1], image.shape[0], (255, 0, 0))
        cv2.putText(panel_nbjw, "NBJW", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(panel_refined, "refined", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 180, 0), 3)
        cv2.putText(panel_gt, "GT", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        tiled = np.concatenate([panel_nbjw, panel_refined, panel_gt], axis=1)
        cv2.imwrite(str(overlay_dir / f"{frame.video}_{frame.frame:06d}.jpg"), tiled)
        written += 1
    if written < min(args.num_overlays, len(selected)):
        print(f"Overlay generation wrote {written}/{len(selected)} frames.", file=sys.stderr)


def evenly_spaced_frames(frames: Sequence[StateFrame], count: int) -> list[StateFrame]:
    if not frames or count <= 0:
        return []
    if len(frames) <= count:
        return list(frames)
    indices = np.linspace(0, len(frames) - 1, count).round().astype(int)
    seen: set[int] = set()
    out: list[StateFrame] = []
    for idx in indices:
        if int(idx) not in seen:
            seen.add(int(idx))
            out.append(frames[int(idx)])
    return out


def load_h_by_video_frame(zf: zipfile.ZipFile, frames: Sequence[StateFrame]) -> dict[tuple[str, int], Optional[np.ndarray]]:
    wanted_members = {frame.member_id for frame in frames}
    rows_by_member_frame = {(frame.member_id, str(scalar(frame.image_id))): frame for frame in frames}
    out: dict[tuple[str, int], Optional[np.ndarray]] = {}
    for member_id in wanted_members:
        member = f"{member_id}_image.pkl"
        if member not in zf.namelist():
            continue
        df = read_pickle(zf, member)
        for index, record in dataframe_rows(df):
            image_id = str(scalar(record.get("id", index)))
            key = (member_id, image_id)
            frame = rows_by_member_frame.get(key)
            if frame is not None:
                out[(frame.video, frame.frame)] = normalize_homography(record.get("h"))
    return out


def draw_projected_pitch(cv2: Any, image: np.ndarray, h_image_to_pitch: Optional[np.ndarray], color: tuple[int, int, int]) -> None:
    h = normalize_homography(h_image_to_pitch)
    inv = invert_homography(h) if h is not None else None
    if inv is None:
        return
    for endpoints in PITCH_LINE_ENDPOINTS_TEMPLATE.values():
        pts_template = np.asarray(endpoints, dtype=float)[:, :2]
        pts_soccer = transform_template_points_to_soccer(pts_template)
        draw_polyline_from_pitch(cv2, image, inv, pts_soccer, color)
    theta = np.linspace(0, 2.0 * np.pi, 100)
    circle = np.stack(
        [CENTRAL_CIRCLE_RADIUS * np.cos(theta), CENTRAL_CIRCLE_RADIUS * np.sin(theta)],
        axis=1,
    )
    draw_polyline_from_pitch(cv2, image, inv, circle, color)


def draw_polyline_from_pitch(
    cv2: Any,
    image: np.ndarray,
    h_pitch_to_image: np.ndarray,
    pitch_points: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    projected, finite = project_points(h_pitch_to_image, pitch_points)
    if finite.sum() < 2:
        return
    pts = np.round(projected[finite]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)


def draw_gt_lines(
    cv2: Any,
    image: np.ndarray,
    gt_lines: Mapping[str, Any],
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    if not isinstance(gt_lines, Mapping):
        return
    for points in gt_lines.values():
        pts = line_points_to_pixel_array(points, width, height)
        if len(pts) >= 2:
            cv2.polylines(image, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)


def line_points_to_pixel_array(points: Any, width: int, height: int) -> np.ndarray:
    out = []
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        return np.zeros((0, 2), dtype=float)
    for point in points:
        if isinstance(point, Mapping):
            if "x" not in point or "y" not in point:
                continue
            x, y = float(point["x"]), float(point["y"])
        elif isinstance(point, Sequence) and len(point) >= 2:
            x, y = float(point[0]), float(point[1])
        else:
            continue
        if abs(x) <= 2.0 and abs(y) <= 2.0:
            x *= width
            y *= height
        out.append([x, y])
    return np.asarray(out, dtype=float)


def maybe_evaluate_homography_states(args: argparse.Namespace) -> Optional[dict[str, Any]]:
    eval_out = args.eval_out or (args.out_state.parent.parent / "homography_eval.json")
    try:
        from sn_gamestate.structured_calibration.metrics import homography_accuracy_eval
    except Exception as exc:
        print(f"Homography evaluator skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    videos = [normalize_video(v) for v in args.videos]
    if not videos:
        return None
    try:
        eval_data_root = split_root(args.dataset_root, args.split)
        source_h = load_homographies_for_eval(args.source_state, videos)
        refined_h = load_homographies_for_eval(args.out_state, videos)
        result = {
            "threshold_px": args.eval_threshold,
            "L0": homography_accuracy_eval(
                source_h,
                eval_data_root,
                videos,
                threshold=args.eval_threshold,
                nproc=args.nproc,
                stride=1,
            ),
            "L2": homography_accuracy_eval(
                refined_h,
                eval_data_root,
                videos,
                threshold=args.eval_threshold,
                nproc=args.nproc,
                stride=1,
            ),
        }
    except Exception as exc:
        print(f"Homography evaluator failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    with eval_out.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return {"path": str(eval_out), "result": result}


def load_homographies_for_eval(state_pklz: Path, videos: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    wanted = {normalize_video(v) for v in videos}
    out: dict[str, dict[str, np.ndarray]] = {video: {} for video in wanted}
    with zipfile.ZipFile(state_pklz, "r") as zf:
        for member in zf.namelist():
            kind, member_id = member_kind(member)
            if kind != "image" or member_id is None:
                continue
            df = read_pickle(zf, member)
            for index, record in dataframe_rows(df):
                video = infer_video_from_record(record, member_id)
                if video is None and len(wanted) == 1:
                    video = next(iter(wanted))
                if video is None:
                    continue
                video = normalize_video(video)
                if video not in wanted:
                    continue
                h = normalize_homography(record.get("h"))
                if h is None:
                    continue
                image_id = str(scalar(record.get("id", index)))
                out.setdefault(video, {})[image_id] = h
    return out


def main() -> None:
    args = parse_args()
    if args.infer_cache_only:
        frames = collect_state_frames(args.source_state, args.dataset_root, args.split, args.videos, args.max_frames)
        run_b2p_inference_to_cache(frames, args)
        print(json.dumps({"cache_frames": len(frames), "cache_dir": str(args.cache_dir)}, ensure_ascii=False))
        return
    try:
        summary = refine_state(args)
    except StateDiagnosisError as exc:
        print(json.dumps({"ok": False, "diagnosis": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(2) from None
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


if __name__ == "__main__":
    main()
