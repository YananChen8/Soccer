"""Atomic track-level role, team, and jersey evaluation for SoccerNet GSR.

This script intentionally stays outside the official GS-HOTA TrackEval metric.
It uses image-space bbox IoU only to decide object matches, aggregates those
frame matches into one fixed GT-track to predicted-track assignment, then scores
attributes on matched tracks.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROLE_CLASSES = ("player", "goalkeeper", "referee", "other")
TEAM_CLASSES = ("left", "right")
MISSING_LABEL = "__missing__"
UNKNOWN_LABEL = "__unknown__"


@dataclass(frozen=True)
class Detection:
    video: str
    frame: int
    track_id: str
    bbox_ltwh: Tuple[float, float, float, float]
    pitch_xy: Optional[Tuple[float, float]] = None
    role: Optional[str] = None
    team: Optional[str] = None
    jersey: Optional[str] = None
    confidence: float = 1.0


@dataclass(frozen=True)
class FrameMatch:
    video: str
    frame: int
    gt_track: str
    pred_track: str
    iou: float


@dataclass(frozen=True)
class TrackMatch:
    video: str
    gt_track: str
    pred_track: str
    frame_matches: int
    iou_sum: float


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if type(value).__name__ == "NAType":
        return True
    try:
        if bool(value != value):
            return True
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null", "nan", "na", "n/a", "<na>"}
    return False


def first_not_none(*values: Optional[Any]) -> Optional[Any]:
    for value in values:
        if value is not None:
            return value
    return None


def scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def normalize_video(value: Any) -> str:
    text = str(scalar(value)).strip()
    match = re.search(r"SNGS[-_]?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"SNGS-{int(match.group(1)):03d}"
    if text.isdigit():
        return f"SNGS-{int(text):03d}"
    return text


def normalize_track_id(value: Any) -> str:
    value = scalar(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def normalize_role(value: Any) -> Optional[str]:
    if is_missing(value):
        return None
    text = str(scalar(value)).strip().lower()
    aliases = {
        "p": "player",
        "player": "player",
        "goalie": "goalkeeper",
        "goal keeper": "goalkeeper",
        "goalkeeper": "goalkeeper",
        "gk": "goalkeeper",
        "ref": "referee",
        "referee": "referee",
        "other": "other",
    }
    return aliases.get(text, text)


def normalize_team(value: Any) -> Optional[str]:
    if is_missing(value):
        return None
    text = str(scalar(value)).strip().lower()
    if text in TEAM_CLASSES:
        return text
    return text


def normalize_jersey(value: Any) -> Optional[str]:
    if is_missing(value):
        return None
    value = scalar(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return str(int(value))
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text.lower() in {"-1", "unknown", "no number", "no jersey", "none", "null"}:
        return None
    if re.fullmatch(r"\d+(\.0+)?", text):
        return str(int(float(text)))
    return text


def parse_frame(value: Any) -> Optional[int]:
    if is_missing(value):
        return None
    value = scalar(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return None
        text = str(int(value))
    else:
        text = Path(str(value)).stem
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    token = nums[-1]
    if len(token) > 6:
        token = token[-6:]
    return int(token)


def image_frame_map(data: Mapping[str, Any]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for image in data.get("images", []) or []:
        image_id = image.get("id", image.get("image_id"))
        frame = image.get("frame")
        if is_missing(frame):
            frame = image.get("file_name", image.get("file_path"))
        parsed = parse_frame(frame if not is_missing(frame) else image_id)
        if image_id is not None and parsed is not None:
            mapping[str(scalar(image_id))] = parsed
    return mapping


def frame_from_annotation(ann: Mapping[str, Any], frames: Mapping[str, int]) -> Optional[int]:
    image_id = ann.get("image_id")
    if image_id is not None and str(scalar(image_id)) in frames:
        return frames[str(scalar(image_id))]
    for key in ("frame", "file_name", "file_path", "image_id"):
        parsed = parse_frame(ann.get(key))
        if parsed is not None:
            return parsed
    return None


def bbox_from_value(value: Any, bbox_format: str = "ltwh") -> Optional[Tuple[float, float, float, float]]:
    if is_missing(value):
        return None
    value = scalar(value)
    if isinstance(value, Mapping):
        if {"x", "y", "w", "h"}.issubset(value):
            return (float(value["x"]), float(value["y"]), float(value["w"]), float(value["h"]))
        if {"left", "top", "width", "height"}.issubset(value):
            return (
                float(value["left"]),
                float(value["top"]),
                float(value["width"]),
                float(value["height"]),
            )
        if {"x1", "y1", "x2", "y2"}.issubset(value):
            x1, y1, x2, y2 = (float(value[k]) for k in ("x1", "y1", "x2", "y2"))
            return (x1, y1, x2 - x1, y2 - y1)
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    x0, y0, a, b = (float(v) for v in value[:4])
    if bbox_format == "ltrb":
        return (x0, y0, a - x0, b - y0)
    return (x0, y0, a, b)


def bbox_from_record(record: Mapping[str, Any], bbox_format: str) -> Optional[Tuple[float, float, float, float]]:
    candidates = (
        ("bbox_ltwh", "ltwh"),
        ("track_bbox_ltwh", "ltwh"),
        ("track_bbox_kf_ltwh", "ltwh"),
        ("track_bbox_pred_kf_ltwh", "ltwh"),
        ("bbox", bbox_format),
        ("bbox_image", bbox_format),
        ("bbox_ltrb", "ltrb"),
        ("track_bbox_ltrb", "ltrb"),
    )
    for key, fmt in candidates:
        if key in record:
            bbox = bbox_from_value(record[key], fmt)
            if bbox is not None and bbox[2] > 0 and bbox[3] > 0:
                return bbox
    return None


def pitch_xy_from_value(value: Any) -> Optional[Tuple[float, float]]:
    if is_missing(value):
        return None
    value = scalar(value)
    if isinstance(value, Mapping):
        if {"x_bottom_middle", "y_bottom_middle"}.issubset(value):
            return (float(value["x_bottom_middle"]), float(value["y_bottom_middle"]))
        if {"x", "y"}.issubset(value):
            return (float(value["x"]), float(value["y"]))
        if {"X", "Y"}.issubset(value):
            return (float(value["X"]), float(value["Y"]))
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    return None


def pitch_xy_from_record(record: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    for key in ("bbox_pitch", "pitch", "pitch_xy", "position", "track_bbox_pitch"):
        if key in record:
            pitch_xy = pitch_xy_from_value(record[key])
            if pitch_xy is not None:
                return pitch_xy
    return None


def attrs_from_record(record: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    attrs_value = record.get("attributes")
    attrs = attrs_value if isinstance(attrs_value, Mapping) else {}
    role = first_present(record, attrs, ("role", "role_detection"))
    team = first_present(record, attrs, ("team", "team_detection"))
    jersey = first_present(
        record,
        attrs,
        ("jersey_number", "jersey", "jersey_number_detection", "jersey_detection"),
    )
    return normalize_role(role), normalize_team(team), normalize_jersey(jersey)


def first_present(
    record: Mapping[str, Any],
    attrs: Mapping[str, Any],
    keys: Sequence[str],
) -> Any:
    for key in keys:
        if key in record and not is_missing(record[key]):
            return record[key]
        if key in attrs and not is_missing(attrs[key]):
            return attrs[key]
    return None


def should_keep_record(record: Mapping[str, Any], ignore_ball: bool) -> bool:
    supercategory = str(record.get("supercategory", "object")).lower()
    if supercategory == "pitch":
        return False
    role, _, _ = attrs_from_record(record)
    if ignore_ball and role == "ball":
        return False
    category_id = record.get("category_id")
    if category_id is not None:
        try:
            if int(float(category_id)) != 1 and role not in ROLE_CLASSES:
                return False
        except Exception:
            pass
    return True


def load_json_detections(
    json_path: Path,
    video: str,
    bbox_format: str,
    ignore_ball: bool,
    detections_key: Optional[str] = None,
) -> List[Detection]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        records = data
        frames: Dict[str, int] = {}
    elif isinstance(data, Mapping):
        frames = image_frame_map(data)
        if detections_key is None:
            detections_key = "predictions" if "predictions" in data else "annotations"
        records = data.get(detections_key, []) or []
    else:
        raise ValueError(f"Unsupported JSON structure in {json_path}")

    detections: List[Detection] = []
    for record in records:
        if not isinstance(record, Mapping) or not should_keep_record(record, ignore_ball):
            continue
        frame = frame_from_annotation(record, frames)
        track_id = record.get("track_id", record.get("id"))
        bbox = bbox_from_record(record, bbox_format)
        if frame is None or track_id is None or bbox is None:
            continue
        role, team, jersey = attrs_from_record(record)
        pitch_xy = pitch_xy_from_record(record)
        conf = record.get("bbox_conf", record.get("confidence", 1.0))
        detections.append(
            Detection(
                video=video,
                frame=frame,
                track_id=normalize_track_id(track_id),
                bbox_ltwh=bbox,
                pitch_xy=pitch_xy,
                role=role,
                team=team,
                jersey=jersey,
                confidence=float(scalar(conf) if not is_missing(conf) else 1.0),
            )
        )
    return detections


def discover_gt_videos(dataset_root: Path, videos: Sequence[str]) -> List[str]:
    if videos:
        return [normalize_video(video) for video in videos]
    return sorted(path.name for path in dataset_root.glob("SNGS-*") if path.is_dir())


def resolve_dataset_root(dataset_root: Path, split: str) -> Path:
    split_root = dataset_root / split
    if split_root.is_dir():
        return split_root
    return dataset_root


def load_gt_dataset(
    dataset_root: Path,
    split: str,
    videos: Sequence[str],
    bbox_format: str,
    ignore_ball: bool,
) -> Dict[str, List[Detection]]:
    root = resolve_dataset_root(dataset_root, split)
    loaded: Dict[str, List[Detection]] = {}
    for video in discover_gt_videos(root, videos):
        labels = root / video / "Labels-GameState.json"
        if not labels.exists():
            raise FileNotFoundError(f"Missing GT label file: {labels}")
        loaded[video] = load_json_detections(labels, video, bbox_format, ignore_ball, "annotations")
    return loaded


def parse_state_video_map(items: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Bad --state-video-map item: {item}")
        key, value = item.split(":", 1)
        mapping[key.strip()] = normalize_video(value)
    return mapping


def video_from_text(value: Any) -> Optional[str]:
    if is_missing(value):
        return None
    match = re.search(r"SNGS[-_]?(\d+)", str(scalar(value)), flags=re.IGNORECASE)
    if match:
        return f"SNGS-{int(match.group(1)):03d}"
    return None


def dataframe_rows(df: Any) -> Iterable[Tuple[Any, Dict[str, Any]]]:
    for index, row in df.iterrows():
        record = row.to_dict()
        record.setdefault("_index", index)
        yield index, record


def load_state_detections(
    state_pklz: Path,
    videos: Sequence[str],
    bbox_format: str,
    ignore_ball: bool,
    state_video_map: Mapping[str, str],
) -> Dict[str, List[Detection]]:
    want = set(normalize_video(video) for video in videos) if videos else None
    loaded: Dict[str, List[Detection]] = defaultdict(list)

    with zipfile.ZipFile(state_pklz, "r") as zf:
        members = [
            name
            for name in zf.namelist()
            if name.endswith(".pkl") and not name.endswith("_image.pkl")
        ]
        for member in members:
            stem = Path(member).stem
            image_member = f"{stem}_image.pkl"
            image_info = load_state_image_info(zf, image_member)
            with zf.open(member, "r") as fp:
                df = pickle.load(fp)
            for _, record in dataframe_rows(df):
                if not should_keep_record(record, ignore_ball):
                    continue
                image_id = record.get("image_id")
                info = image_info.get(str(scalar(image_id)), {}) if image_id is not None else {}
                video = (
                    video_from_text(record.get("video"))
                    or video_from_text(record.get("video_name"))
                    or video_from_text(record.get("file_path"))
                    or info.get("video")
                    or state_video_map.get(stem)
                )
                if video is None:
                    if want and len(want) == 1:
                        video = next(iter(want))
                    else:
                        raise ValueError(
                            "Cannot infer video name for state member "
                            f"{member}; pass --state-video-map {stem}:SNGS-xxx"
                        )
                if want and video not in want:
                    continue
                frame = first_not_none(
                    info.get("frame"),
                    parse_frame(record.get("image_id")),
                    parse_frame(record.get("frame")),
                    parse_frame(record.get("file_path")),
                )
                if frame is None:
                    raise ValueError(f"Cannot infer frame for state member {member}, image_id={image_id}")
                bbox = bbox_from_record(record, bbox_format)
                track_id = record.get("track_id")
                if bbox is None or track_id is None or is_missing(track_id):
                    continue
                role, team, jersey = attrs_from_record(record)
                pitch_xy = pitch_xy_from_record(record)
                conf = record.get("bbox_conf", record.get("track_bbox_conf", 1.0))
                loaded[video].append(
                    Detection(
                        video=video,
                        frame=frame,
                        track_id=normalize_track_id(track_id),
                        bbox_ltwh=bbox,
                        pitch_xy=pitch_xy,
                        role=role,
                        team=team,
                        jersey=jersey,
                        confidence=float(scalar(conf) if not is_missing(conf) else 1.0),
                    )
                )
    return dict(loaded)


def load_state_image_info(zf: zipfile.ZipFile, image_member: str) -> Dict[str, Dict[str, Any]]:
    if image_member not in zf.namelist():
        return {}
    with zf.open(image_member, "r") as fp:
        image_df = pickle.load(fp)
    info: Dict[str, Dict[str, Any]] = {}
    for index, record in dataframe_rows(image_df):
        image_id = record.get("id", index)
        video = (
            video_from_text(record.get("video"))
            or video_from_text(record.get("video_name"))
            or video_from_text(record.get("name"))
            or video_from_text(record.get("file_path"))
        )
        frame = first_not_none(
            parse_frame(record.get("file_path")),
            parse_frame(record.get("id")),
            parse_frame(record.get("image_id")),
            parse_frame(record.get("frame")),
        )
        info[str(scalar(image_id))] = {"video": video, "frame": frame}
    return info


def load_pred_json_dir(
    pred_dir: Path,
    videos: Sequence[str],
    bbox_format: str,
    ignore_ball: bool,
) -> Dict[str, List[Detection]]:
    want = set(normalize_video(video) for video in videos) if videos else None
    loaded: Dict[str, List[Detection]] = {}
    for json_path in sorted(pred_dir.glob("*.json")):
        video = normalize_video(json_path.stem)
        if want and video not in want:
            continue
        loaded[video] = load_json_detections(json_path, video, bbox_format, ignore_ball)
    if want:
        missing = sorted(want - set(loaded))
        if missing:
            raise FileNotFoundError(f"Missing prediction json files for: {', '.join(missing)}")
    return loaded


def dedupe_detections(detections: Iterable[Detection]) -> List[Detection]:
    best: Dict[Tuple[str, int, str], Detection] = {}
    for det in detections:
        key = (det.video, det.frame, det.track_id)
        prev = best.get(key)
        if prev is None or det.confidence > prev.confidence:
            best[key] = det
    return list(best.values())


def iou_ltwh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    if union <= 0:
        return 0.0
    return inter / union


def linear_assignment_max(scores: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    if not scores or not scores[0]:
        return []
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        max_score = max(max(row) for row in scores)
        cost = [[max_score - value for value in row] for row in scores]
        rows, cols = linear_sum_assignment(cost)
        return [(int(r), int(c)) for r, c in zip(rows, cols)]
    except Exception:
        return hungarian_max(scores)


def hungarian_max(scores: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    n_rows = len(scores)
    n_cols = len(scores[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        return []
    if n_rows > n_cols:
        transposed = [[scores[i][j] for i in range(n_rows)] for j in range(n_cols)]
        return [(col, row) for row, col in hungarian_max(transposed)]

    max_score = max(max(row) for row in scores)
    cost = [[max_score - value for value in row] for row in scores]
    assignment = hungarian_min(cost)
    return [(row, col) for row, col in enumerate(assignment) if col >= 0]


def hungarian_min(cost: Sequence[Sequence[float]]) -> List[int]:
    n = len(cost)
    m = len(cost[0]) if n else 0
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def group_by_frame(detections: Iterable[Detection]) -> Dict[int, List[Detection]]:
    grouped: Dict[int, List[Detection]] = defaultdict(list)
    for det in detections:
        grouped[det.frame].append(det)
    return grouped


def match_video_frames(
    video: str,
    gt_detections: Sequence[Detection],
    pred_detections: Sequence[Detection],
    iou_threshold: float,
) -> List[FrameMatch]:
    gt_by_frame = group_by_frame(gt_detections)
    pred_by_frame = group_by_frame(pred_detections)
    matches: List[FrameMatch] = []
    for frame in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gt_frame = gt_by_frame.get(frame, [])
        pred_frame = pred_by_frame.get(frame, [])
        if not gt_frame or not pred_frame:
            continue
        scores = [[iou_ltwh(gt.bbox_ltwh, pred.bbox_ltwh) for pred in pred_frame] for gt in gt_frame]
        for gt_idx, pred_idx in linear_assignment_max(scores):
            iou = scores[gt_idx][pred_idx]
            if iou >= iou_threshold:
                matches.append(
                    FrameMatch(
                        video=video,
                        frame=frame,
                        gt_track=gt_frame[gt_idx].track_id,
                        pred_track=pred_frame[pred_idx].track_id,
                        iou=iou,
                    )
                )
    return matches


def match_tracks(frame_matches: Sequence[FrameMatch], min_track_matches: int) -> List[TrackMatch]:
    by_video: Dict[str, List[FrameMatch]] = defaultdict(list)
    for match in frame_matches:
        by_video[match.video].append(match)

    track_matches: List[TrackMatch] = []
    for video, matches in sorted(by_video.items()):
        gt_tracks = sorted({match.gt_track for match in matches}, key=natural_key)
        pred_tracks = sorted({match.pred_track for match in matches}, key=natural_key)
        gt_index = {track: idx for idx, track in enumerate(gt_tracks)}
        pred_index = {track: idx for idx, track in enumerate(pred_tracks)}
        counts = [[0.0 for _ in pred_tracks] for _ in gt_tracks]
        iou_sums = [[0.0 for _ in pred_tracks] for _ in gt_tracks]
        for match in matches:
            i = gt_index[match.gt_track]
            j = pred_index[match.pred_track]
            counts[i][j] += 1.0
            iou_sums[i][j] += match.iou

        scores = [
            [counts[i][j] + 1e-6 * iou_sums[i][j] for j in range(len(pred_tracks))]
            for i in range(len(gt_tracks))
        ]
        for i, j in linear_assignment_max(scores):
            count = int(counts[i][j])
            if count >= min_track_matches:
                track_matches.append(
                    TrackMatch(
                        video=video,
                        gt_track=gt_tracks[i],
                        pred_track=pred_tracks[j],
                        frame_matches=count,
                        iou_sum=iou_sums[i][j],
                    )
                )
    return track_matches


def natural_key(value: str) -> Tuple[Any, ...]:
    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def track_key(det: Detection) -> Tuple[str, str]:
    return det.video, det.track_id


def collect_track_values(detections: Iterable[Detection]) -> Dict[Tuple[str, str], Dict[str, Counter]]:
    values: Dict[Tuple[str, str], Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for det in detections:
        key = track_key(det)
        for attr in ("role", "team", "jersey"):
            value = getattr(det, attr)
            if value is not None:
                values[key][attr][value] += 1
    return values


def majority_value(counter: Counter) -> Optional[str]:
    if not counter:
        return None
    return sorted(counter.items(), key=lambda item: (-item[1], natural_key(str(item[0]))))[0][0]


def label_for_role(value: Optional[str]) -> str:
    if value is None:
        return MISSING_LABEL
    if value in ROLE_CLASSES:
        return value
    return UNKNOWN_LABEL


def compute_role_metrics(pairs: Sequence[Tuple[Optional[str], Optional[str]]]) -> Dict[str, Any]:
    matrix: Dict[str, Dict[str, int]] = {
        cls: {pred: 0 for pred in list(ROLE_CLASSES) + [MISSING_LABEL, UNKNOWN_LABEL]}
        for cls in ROLE_CLASSES
    }
    for gt_role, pred_role in pairs:
        if gt_role not in ROLE_CLASSES:
            continue
        pred_label = label_for_role(pred_role)
        matrix[gt_role][pred_label] += 1

    per_class = {}
    f1_values = []
    for cls in ROLE_CLASSES:
        tp = matrix[cls][cls]
        fp = sum(matrix[gt][cls] for gt in ROLE_CLASSES if gt != cls)
        fn = sum(count for pred, count in matrix[cls].items() if pred != cls)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": sum(matrix[cls].values()),
        }
        f1_values.append(f1)
    return {
        "macro_f1": sum(f1_values) / len(ROLE_CLASSES),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "total": sum(sum(row.values()) for row in matrix.values()),
    }


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_accuracy(pairs: Sequence[Tuple[Optional[str], Optional[str]]]) -> Dict[str, Any]:
    total = len(pairs)
    correct = sum(1 for gt_value, pred_value in pairs if gt_value == pred_value)
    return {
        "accuracy": safe_div(correct, total),
        "correct": correct,
        "total": total,
    }


def evaluate_attributes(
    gt_by_video: Mapping[str, Sequence[Detection]],
    pred_by_video: Mapping[str, Sequence[Detection]],
    iou_threshold: float,
    min_track_matches: int,
    include_missing_gt_jersey: bool,
) -> Dict[str, Any]:
    videos = sorted(set(gt_by_video) & set(pred_by_video))
    all_frame_matches: List[FrameMatch] = []
    per_video_frame_counts: Dict[str, Dict[str, int]] = {}
    for video in videos:
        matches = match_video_frames(video, gt_by_video[video], pred_by_video[video], iou_threshold)
        all_frame_matches.extend(matches)
        per_video_frame_counts[video] = {
            "gt_detections": len(gt_by_video[video]),
            "pred_detections": len(pred_by_video[video]),
            "frame_matches": len(matches),
        }

    track_matches = match_tracks(all_frame_matches, min_track_matches)
    gt_values = collect_track_values(det for detections in gt_by_video.values() for det in detections)
    pred_values = collect_track_values(det for detections in pred_by_video.values() for det in detections)

    role_pairs: List[Tuple[Optional[str], Optional[str]]] = []
    team_pairs: List[Tuple[Optional[str], Optional[str]]] = []
    jersey_pairs: List[Tuple[Optional[str], Optional[str]]] = []
    matched_rows = []

    for match in track_matches:
        gt_key = (match.video, match.gt_track)
        pred_key = (match.video, match.pred_track)
        gt_role = majority_value(gt_values[gt_key]["role"])
        pred_role = majority_value(pred_values[pred_key]["role"])
        gt_team = majority_value(gt_values[gt_key]["team"])
        pred_team = majority_value(pred_values[pred_key]["team"])
        gt_jersey = majority_value(gt_values[gt_key]["jersey"])
        pred_jersey = majority_value(pred_values[pred_key]["jersey"])

        if gt_role in ROLE_CLASSES:
            role_pairs.append((gt_role, pred_role))
        if gt_team in TEAM_CLASSES:
            team_pairs.append((gt_team, pred_team))
        if include_missing_gt_jersey or gt_jersey is not None:
            jersey_pairs.append((gt_jersey, pred_jersey))

        matched_rows.append(
            {
                "video": match.video,
                "gt_track": match.gt_track,
                "pred_track": match.pred_track,
                "frame_matches": match.frame_matches,
                "mean_iou": safe_div(match.iou_sum, match.frame_matches),
                "gt_role": gt_role,
                "pred_role": pred_role,
                "gt_team": gt_team,
                "pred_team": pred_team,
                "gt_jersey": gt_jersey,
                "pred_jersey": pred_jersey,
            }
        )

    role = compute_role_metrics(role_pairs)
    team = compute_accuracy(team_pairs)
    jersey = compute_accuracy(jersey_pairs)

    per_video: Dict[str, Any] = {}
    for video in videos:
        video_matches = [row for row in matched_rows if row["video"] == video]
        video_role_pairs = [
            (row["gt_role"], row["pred_role"])
            for row in video_matches
            if row["gt_role"] in ROLE_CLASSES
        ]
        video_team_pairs = [
            (row["gt_team"], row["pred_team"])
            for row in video_matches
            if row["gt_team"] in TEAM_CLASSES
        ]
        video_jersey_pairs = [
            (row["gt_jersey"], row["pred_jersey"])
            for row in video_matches
            if include_missing_gt_jersey or row["gt_jersey"] is not None
        ]
        per_video[video] = {
            **per_video_frame_counts[video],
            "matched_tracks": len(video_matches),
            "RoleMacroF1": compute_role_metrics(video_role_pairs)["macro_f1"],
            "TeamTrackAccuracy": compute_accuracy(video_team_pairs)["accuracy"],
            "JerseyTrackExactAccuracy": compute_accuracy(video_jersey_pairs)["accuracy"],
        }

    return {
        "summary": {
            "RoleMacroF1": role["macro_f1"],
            "TeamTrackAccuracy": team["accuracy"],
            "JerseyTrackExactAccuracy": jersey["accuracy"],
            "matched_tracks": len(track_matches),
            "frame_matches": len(all_frame_matches),
            "videos": len(videos),
        },
        "role": role,
        "team": team,
        "jersey": jersey,
        "per_video": per_video,
        "matched_tracks": matched_rows,
    }


def print_report(result: Mapping[str, Any]) -> None:
    summary = result["summary"]
    print("Atomic attribute metrics")
    print(f"  RoleMacroF1:              {summary['RoleMacroF1'] * 100:.3f}")
    print(f"  TeamTrackAccuracy:        {summary['TeamTrackAccuracy'] * 100:.3f}")
    print(f"  JerseyTrackExactAccuracy: {summary['JerseyTrackExactAccuracy'] * 100:.3f}")
    print(f"  matched_tracks:           {summary['matched_tracks']}")
    print(f"  frame_matches:            {summary['frame_matches']}")
    print("")
    print("Role per-class F1")
    for cls in ROLE_CLASSES:
        item = result["role"]["per_class"][cls]
        print(
            f"  {cls:10s} f1={item['f1'] * 100:7.3f} "
            f"p={item['precision'] * 100:7.3f} r={item['recall'] * 100:7.3f} "
            f"support={item['support']}"
        )
    print("")
    print(
        "Team correct/total: "
        f"{result['team']['correct']}/{result['team']['total']}"
    )
    print(
        "Jersey correct/total: "
        f"{result['jersey']['correct']}/{result['jersey']['total']}"
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if hasattr(value, "item"):
        return json_ready(value.item())
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/SoccerNetGS"))
    parser.add_argument("--split", default="valid")
    parser.add_argument("--videos", nargs="*", default=[])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--state-pklz", type=Path)
    source.add_argument("--pred-dir", type=Path)
    parser.add_argument("--state-video-map", nargs="*", default=[])
    parser.add_argument("--bbox-format", choices=("ltwh", "ltrb"), default="ltwh")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-track-matches", type=int, default=1)
    parser.add_argument("--include-missing-gt-jersey", action="store_true")
    parser.add_argument("--keep-ball", action="store_true")
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ignore_ball = not args.keep_ball
    videos = [normalize_video(video) for video in args.videos]
    gt_by_video = load_gt_dataset(
        args.dataset_root,
        args.split,
        videos,
        args.bbox_format,
        ignore_ball,
    )
    if args.state_pklz:
        pred_by_video = load_state_detections(
            args.state_pklz,
            videos,
            args.bbox_format,
            ignore_ball,
            parse_state_video_map(args.state_video_map),
        )
    else:
        pred_by_video = load_pred_json_dir(args.pred_dir, videos, args.bbox_format, ignore_ball)

    if not args.no_dedupe:
        pred_by_video = {
            video: dedupe_detections(detections)
            for video, detections in pred_by_video.items()
        }

    missing_pred = sorted(set(gt_by_video) - set(pred_by_video))
    if missing_pred:
        raise FileNotFoundError(f"Missing predictions for: {', '.join(missing_pred)}")

    result = evaluate_attributes(
        gt_by_video,
        pred_by_video,
        args.iou_threshold,
        args.min_track_matches,
        args.include_missing_gt_jersey,
    )
    result = {
        "config": {
            "dataset_root": str(args.dataset_root),
            "split": args.split,
            "videos": videos or sorted(gt_by_video),
            "source": str(args.state_pklz or args.pred_dir),
            "bbox_format": args.bbox_format,
            "iou_threshold": args.iou_threshold,
            "min_track_matches": args.min_track_matches,
            "include_missing_gt_jersey": args.include_missing_gt_jersey,
            "dedupe": not args.no_dedupe,
            "ignore_ball": ignore_ball,
        },
        **result,
    }
    print_report(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(json_ready(result), f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
