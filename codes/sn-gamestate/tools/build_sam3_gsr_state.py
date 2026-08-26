#!/usr/bin/env python3
"""Build TrackLab/SoccerNetGS state files from SAM3 video tracking.

The script runs SAM3 or SAM3.1 on SoccerNetGS frame folders and writes a
TrackLab-compatible ``sn-gamestate.pklz``. It is intentionally standalone:
it writes the same zip/pickle layout as ``TrackerState.save()`` without
requiring Hydra or the SoccerNetGS dataset wrapper at generation time.

Typical use from ``codes/sn-gamestate``:

    python tools/build_sam3_gsr_state.py \
      --dataset-root datasets/SoccerNetGS \
      --split valid \
      --videos SNGS-021 \
      --sam3-root ../sam3_official \
      --version sam3 \
      --mode native \
      --out outputs/gsr/sam3_native_021/states/sn-gamestate.pklz

    python tools/build_sam3_gsr_state.py \
      --dataset-root datasets/SoccerNetGS \
      --split valid \
      --videos SNGS-021 \
      --sam3-root ../sam3_official \
      --version sam3 \
      --mode periodic \
      --recondition-every 50 \
      --out outputs/gsr/sam3_periodic50_021/states/sn-gamestate.pklz
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import pickle
import re
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LOG = logging.getLogger("build_sam3_gsr_state")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_PROMPTS = ["player"]
DETECTION_COLUMNS = [
    "id",
    "video_id",
    "image_id",
    "category_id",
    "bbox_ltwh",
    "bbox_conf",
    "track_id",
    "track_bbox_ltwh",
    "track_bbox_conf",
    "role_detection",
    "role_confidence",
    "role",
    "team_color_hint",
    "sam3_prompt",
    "sam3_prompt_index",
    "sam3_source_obj_id",
    "sam3_chunk_index",
    "sam3_chunk_start_frame",
    "sam3_chunk_end_frame",
    "sam3_recondition_every",
    "sam3_prompt_refresh_every",
]
IMAGE_COLUMNS = ["video_id", "file_path", "frame"]


@dataclass(frozen=True)
class FrameRef:
    video: str
    video_id: Any
    image_id: int
    frame: int
    sam3_index: int
    file_path: Path


@dataclass
class MetadataIndex:
    video_to_id: Dict[str, Any]
    image_rows_by_video_frame: Dict[Tuple[str, int], Dict[str, Any]]
    next_image_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 native/periodic tracking and save TrackLab pklz state."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/SoccerNetGS"),
        help="SoccerNetGS root. May contain split folders or be the split folder itself.",
    )
    parser.add_argument("--split", default="valid", help="Dataset split, e.g. valid/test.")
    parser.add_argument(
        "--videos",
        nargs="+",
        default=None,
        help="Video names such as SNGS-021. Defaults to all SNGS-* folders in the split.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output pklz path, usually outputs/.../states/sn-gamestate.pklz.",
    )
    parser.add_argument(
        "--sam3-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "sam3_official",
        help="Path to the checked-out SAM3 package root.",
    )
    parser.add_argument("--version", choices=["sam3", "sam3.1"], default="sam3")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["native", "periodic"],
        default="native",
        help="Naming/validation mode. Native disables internal periodic reconditioning by default.",
    )
    parser.add_argument(
        "--recondition-every",
        type=int,
        default=None,
        help=(
            "Internal SAM3 recondition interval. Defaults to 0 in native mode and 50 "
            "in periodic mode. Set explicitly for periodic60/70/75."
        ),
    )
    parser.add_argument(
        "--prompt-refresh-every",
        type=int,
        default=0,
        help="Externally add the text prompt every N frames before propagation. 0 means frame 0 only.",
    )
    parser.add_argument(
        "--prompt-frames",
        type=int,
        nargs="*",
        default=None,
        help="Explicit prompt frames. Overrides --prompt-refresh-every when provided.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=DEFAULT_PROMPTS,
        help="Text prompts. Multiple prompts are run in separate sessions and merged.",
    )
    parser.add_argument(
        "--prompt-team-colors",
        nargs="*",
        default=None,
        help=(
            "Optional color hints aligned with --prompts. These populate team_color_hint "
            "for the color_prompt team module."
        ),
    )
    parser.add_argument(
        "--metadata-state",
        type=Path,
        default=None,
        help="Optional existing TrackLab pklz to align video_id/image_id with prior states.",
    )
    parser.add_argument(
        "--video-id-map",
        type=Path,
        default=None,
        help="Optional JSON mapping from video name to numeric video_id.",
    )
    parser.add_argument(
        "--preserve-metadata-file-path",
        action="store_true",
        help="When --metadata-state is used, keep image file_path from that state.",
    )
    parser.add_argument("--nframes", type=int, default=-1, help="Limit frames per video.")
    parser.add_argument("--start-frame", type=int, default=0, help="Default first prompt frame.")
    parser.add_argument(
        "--propagation-direction",
        choices=["forward", "backward", "both"],
        default="forward",
    )
    parser.add_argument(
        "--output-prob-thresh",
        type=float,
        default=0.5,
        help="Output probability threshold passed to SAM3 prompt/propagation calls.",
    )
    parser.add_argument("--min-mask-area", type=int, default=8)
    parser.add_argument(
        "--max-mask-area-frac",
        type=float,
        default=0.35,
        help="Drop masks covering more than this fraction of the frame. <=0 disables.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.85,
        help="Per-frame cross-prompt duplicate suppression IoU. <=0 disables.",
    )
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--id-offset", type=int, default=1_000_000)
    parser.add_argument("--chunk-id-offset", type=int, default=10_000)
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=0,
        help="Run each video in independent frame chunks to cap SAM3 memory. 0 means full video.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=0,
        help="Number of frames overlapped between chunks. Only used with --chunk-frames.",
    )
    parser.add_argument(
        "--stitch-chunks",
        action="store_true",
        help="Merge track ids across overlapped chunks using image-space bbox IoU.",
    )
    parser.add_argument("--stitch-iou", type=float, default=0.5)
    parser.add_argument("--max-num-objects", type=int, default=64)
    parser.add_argument("--multiplex-count", type=int, default=16)
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU ids for sam3 multi-GPU.")
    parser.add_argument(
        "--collective-timeout-sec",
        type=int,
        default=1800,
        help=(
            "SAM3 multi-GPU NCCL/Gloo collective timeout. The upstream default is "
            "180 seconds, which can be too short for long videos."
        ),
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warm-up", action="store_true")
    parser.add_argument("--sync-loading-frames", action="store_true")
    parser.add_argument(
        "--offload-video-to-cpu",
        action="store_true",
        help="Keep loaded video frames on CPU to reduce GPU memory use.",
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action="store_true",
        help="Keep SAM3 tracking state on CPU to reduce GPU memory use at lower speed.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Scan inputs and exit before loading SAM3.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def split_root(dataset_root: Path, split: str) -> Path:
    candidate = dataset_root / split
    return candidate if candidate.exists() else dataset_root


def discover_videos(root: Path, videos: Optional[Sequence[str]]) -> List[str]:
    if videos:
        return list(videos)
    found = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("SNGS-"))
    if not found:
        raise FileNotFoundError(f"No SNGS-* video folders found in {root}")
    return found


def infer_video_id_from_name(video: str) -> Optional[str]:
    match = re.fullmatch(r"SNGS-(\d+)", video)
    if not match:
        return None
    return match.group(1)


def frame_dir_for_video(root: Path, video: str) -> Path:
    video_dir = root / video
    img1_dir = video_dir / "img1"
    if img1_dir.exists():
        return img1_dir
    if video_dir.exists():
        return video_dir
    raise FileNotFoundError(f"Cannot find frames for video {video}: {img1_dir} or {video_dir}")


def parse_frame_number(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    stem = Path(text).stem
    if stem.isdigit():
        return int(stem)
    numbers = re.findall(r"\d+", text)
    if not numbers:
        return None
    return int(numbers[-1])


def frame_sort_key(path: Path) -> Tuple[int, str]:
    parsed = parse_frame_number(path.name)
    return (parsed if parsed is not None else 10**12, path.name)


def list_frame_files(frame_dir: Path, nframes: int) -> List[Path]:
    files = sorted(
        [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=frame_sort_key,
    )
    if nframes and nframes > 0:
        files = files[:nframes]
    if not files:
        raise FileNotFoundError(f"No image frames found in {frame_dir}")
    return files


def video_from_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    match = re.search(r"SNGS-\d+", str(value))
    return match.group(0) if match else None


def load_json_mapping(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {str(k): v for k, v in data.items()}


def load_metadata_state(path: Optional[Path]) -> MetadataIndex:
    if path is None:
        return MetadataIndex(video_to_id={}, image_rows_by_video_frame={}, next_image_id=0)
    video_to_id: Dict[str, Any] = {}
    image_rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    next_image_id = 0
    with zipfile.ZipFile(path, "r") as zf:
        for member in zf.namelist():
            if not member.endswith("_image.pkl"):
                continue
            member_stem = Path(member).stem
            member_video_id = member_stem.removesuffix("_image")
            with zf.open(member, "r") as fh:
                df = pickle.load(fh)
            if not isinstance(df, pd.DataFrame):
                continue
            for image_id, row in df.iterrows():
                record = row.to_dict()
                video = (
                    video_from_text(record.get("file_path"))
                    or video_from_text(record.get("video_name"))
                    or video_from_text(record.get("name"))
                )
                frame = parse_frame_number(record.get("frame"))
                if frame is None:
                    frame = parse_frame_number(record.get("file_path"))
                if video is None or frame is None:
                    continue
                video_id = record.get("video_id", member_video_id)
                record["image_id"] = int(image_id)
                record["video_id"] = video_id
                image_rows[(video, frame)] = record
                video_to_id.setdefault(video, video_id)
                next_image_id = max(next_image_id, int(image_id) + 1)
    LOG.info(
        "Loaded metadata alignment from %s: %d videos, %d images",
        path,
        len(video_to_id),
        len(image_rows),
    )
    return MetadataIndex(video_to_id=video_to_id, image_rows_by_video_frame=image_rows, next_image_id=next_image_id)


def prepare_frames(
    root: Path,
    videos: Sequence[str],
    metadata: MetadataIndex,
    video_id_map: Mapping[str, Any],
    nframes: int,
    preserve_metadata_file_path: bool,
) -> Dict[str, List[FrameRef]]:
    next_video_id = 0
    next_image_id = metadata.next_image_id
    frames_by_video: Dict[str, List[FrameRef]] = {}

    used_video_ids = set(video_id_map.values()) | set(metadata.video_to_id.values())
    while next_video_id in used_video_ids:
        next_video_id += 1

    for video in videos:
        if video in video_id_map:
            video_id = video_id_map[video]
        elif video in metadata.video_to_id:
            video_id = metadata.video_to_id[video]
        else:
            inferred_video_id = infer_video_id_from_name(video)
            if inferred_video_id is not None and inferred_video_id not in used_video_ids:
                video_id = inferred_video_id
            else:
                while next_video_id in used_video_ids:
                    next_video_id += 1
                video_id = next_video_id
            used_video_ids.add(video_id)
            if isinstance(video_id, int):
                next_video_id = max(next_video_id, video_id + 1)

        frame_files = list_frame_files(frame_dir_for_video(root, video), nframes)
        frame_refs: List[FrameRef] = []
        for sam3_index, path in enumerate(frame_files):
            parsed_frame = parse_frame_number(path.name)
            frame = parsed_frame if parsed_frame is not None else sam3_index
            meta_row = metadata.image_rows_by_video_frame.get((video, frame))
            if meta_row is not None:
                image_id = int(meta_row["image_id"])
                file_path = Path(meta_row["file_path"]) if preserve_metadata_file_path else path
            else:
                image_id = next_image_id
                next_image_id += 1
                file_path = path
            frame_refs.append(
                FrameRef(
                    video=video,
                    video_id=video_id,
                    image_id=image_id,
                    frame=frame,
                    sam3_index=sam3_index,
                    file_path=file_path,
                )
            )
        frames_by_video[video] = frame_refs
        LOG.info(
            "Prepared %s: video_id=%s, frames=%d, image_id range=%s..%s",
            video,
            video_id,
            len(frame_refs),
            frame_refs[0].image_id,
            frame_refs[-1].image_id,
        )
    return frames_by_video


def prompt_frames_for_video(args: argparse.Namespace, num_frames: int) -> List[int]:
    if args.prompt_frames is not None and len(args.prompt_frames) > 0:
        frames = sorted({f for f in args.prompt_frames if 0 <= f < num_frames})
        return frames or [min(max(args.start_frame, 0), num_frames - 1)]
    if args.prompt_refresh_every and args.prompt_refresh_every > 0:
        frames = list(range(max(args.start_frame, 0), num_frames, args.prompt_refresh_every))
        return frames or [0]
    return [min(max(args.start_frame, 0), num_frames - 1)]


def iter_frame_chunks(
    frame_refs: Sequence[FrameRef],
    chunk_frames: int,
    chunk_overlap: int,
) -> Iterable[Tuple[int, int, int, Sequence[FrameRef]]]:
    num_frames = len(frame_refs)
    if chunk_frames <= 0 or chunk_frames >= num_frames:
        yield 0, 0, num_frames, frame_refs
        return
    if chunk_overlap < 0:
        raise ValueError("--chunk-overlap must be >= 0")
    if chunk_overlap >= chunk_frames:
        raise ValueError("--chunk-overlap must be smaller than --chunk-frames")

    step = chunk_frames - chunk_overlap
    chunk_index = 0
    start = 0
    while start < num_frames:
        end = min(start + chunk_frames, num_frames)
        yield chunk_index, start, end, frame_refs[start:end]
        if end >= num_frames:
            break
        start += step
        chunk_index += 1


def prompt_frames_for_chunk(
    args: argparse.Namespace,
    chunk_start: int,
    chunk_len: int,
    full_len: int,
) -> List[int]:
    if chunk_len <= 0:
        return []
    if args.chunk_frames <= 0:
        return prompt_frames_for_video(args, chunk_len)

    chunk_end = chunk_start + chunk_len
    if args.prompt_frames is not None and len(args.prompt_frames) > 0:
        local_frames = sorted(
            {int(frame) - chunk_start for frame in args.prompt_frames if chunk_start <= int(frame) < chunk_end}
        )
        return local_frames or [0]

    if args.prompt_refresh_every and args.prompt_refresh_every > 0:
        start = max(int(args.start_frame), 0)
        global_frames = range(start, full_len, int(args.prompt_refresh_every))
        local_frames = {int(frame) - chunk_start for frame in global_frames if chunk_start <= int(frame) < chunk_end}
        local_frames.add(0)
        return sorted(local_frames)

    return [0]


def normalize_recondition_every(args: argparse.Namespace) -> int:
    if args.recondition_every is not None:
        return int(args.recondition_every)
    return 50 if args.mode == "periodic" else 0


def ensure_out_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def import_sam3(args: argparse.Namespace):
    sam3_root = args.sam3_root.resolve()
    if not sam3_root.exists():
        raise FileNotFoundError(f"--sam3-root does not exist: {sam3_root}")
    sys.path.insert(0, str(sam3_root))
    os.environ.setdefault("USE_PERFLIB", "1")
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / f"torchinductor_cache_{getpass.getuser()}"),
    )
    import torch
    from sam3 import build_sam3_predictor

    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 video predictor requires CUDA in this checkout.")
    return torch, build_sam3_predictor


def parse_gpus(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_predictor(args: argparse.Namespace, recondition_every: int):
    torch, build_sam3_predictor = import_sam3(args)
    kwargs: Dict[str, Any] = {
        "version": args.version,
        "compile": args.compile,
        "warm_up": args.warm_up,
        "async_loading_frames": not args.sync_loading_frames,
    }
    if args.checkpoint is not None:
        kwargs["checkpoint_path"] = str(args.checkpoint)
    if args.version == "sam3.1":
        kwargs["max_num_objects"] = args.max_num_objects
        kwargs["multiplex_count"] = args.multiplex_count
    else:
        gpus = parse_gpus(args.gpus)
        if gpus is not None:
            if args.collective_timeout_sec > 0:
                os.environ["SAM3_COLLECTIVE_OP_TIMEOUT_SEC"] = str(args.collective_timeout_sec)
            kwargs["gpus_to_use"] = gpus
    LOG.info("Building %s predictor", args.version)
    predictor = build_sam3_predictor(**kwargs)
    set_max_num_objects(predictor, args.max_num_objects)
    set_recondition_every(predictor, recondition_every)
    return torch, predictor


def set_max_num_objects(predictor: Any, limit: int) -> None:
    if limit <= 0:
        return
    targets = []
    if hasattr(predictor, "model"):
        targets.append(predictor.model)
    targets.append(predictor)
    updated = 0
    for target in targets:
        if hasattr(target, "max_num_objects"):
            setattr(target, "max_num_objects", int(limit))
            world_size = int(getattr(target, "world_size", 1) or 1)
            if hasattr(target, "num_obj_for_compile"):
                setattr(target, "num_obj_for_compile", max(1, (int(limit) + world_size - 1) // world_size))
            updated += 1
    if updated == 0:
        LOG.warning("Could not find max_num_objects on predictor/model")
    else:
        LOG.info("Set internal SAM3 max_num_objects to %d", limit)


def set_recondition_every(predictor: Any, interval: int) -> None:
    targets = []
    if hasattr(predictor, "model"):
        targets.append(predictor.model)
    targets.append(predictor)
    updated = 0
    for target in targets:
        if hasattr(target, "recondition_every_nth_frame"):
            setattr(target, "recondition_every_nth_frame", int(interval))
            updated += 1
    if updated == 0:
        LOG.warning("Could not find recondition_every_nth_frame on predictor/model")
    else:
        LOG.info("Set internal SAM3 recondition interval to %d", interval)


def close_predictor(predictor: Any) -> None:
    shutdown = getattr(predictor, "shutdown", None)
    if callable(shutdown):
        shutdown()


def to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and callable(value.cpu):
        return value.cpu().numpy()
    return np.asarray(value)


def denormalize_xywh(box: Sequence[float], width: int, height: int) -> Tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in box]
    if max(abs(x), abs(y), abs(w), abs(h)) <= 2.0:
        x *= width
        w *= width
        y *= height
        h *= height
    x = max(0.0, min(x, float(width - 1)))
    y = max(0.0, min(y, float(height - 1)))
    w = max(0.0, min(w, float(width) - x))
    h = max(0.0, min(h, float(height) - y))
    return (x, y, w, h)


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    if mask.ndim == 3:
        mask = mask[0]
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return None
    ys, xs = np.where(mask_bool)
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max() + 1)
    y1 = float(ys.max() + 1)
    return (x0, y0, x1 - x0, y1 - y0)


def bbox_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2])) * max(0.0, float(box[3]))


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, aw, ah = [float(v) for v in a]
    bx1, by1, bw, bh = [float(v) for v in b]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def read_image_size(path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        return int(width), int(height)
    except Exception:
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image size from {path}")
        height, width = image.shape[:2]
        return int(width), int(height)


def collect_prompt_records(
    predictor: Any,
    frame_dir: Path,
    frame_refs: Sequence[FrameRef],
    prompt: str,
    prompt_index: int,
    team_color_hint: Optional[str],
    prompt_frames: Sequence[int],
    chunk_index: int,
    chunk_start_frame: int,
    chunk_end_frame: int,
    args: argparse.Namespace,
    recondition_every: int,
) -> List[Dict[str, Any]]:
    session = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": str(frame_dir),
            "offload_video_to_cpu": bool(args.offload_video_to_cpu),
            "offload_state_to_cpu": bool(args.offload_state_to_cpu),
        }
    )
    session_id = session["session_id"]
    width, height = read_image_size(frame_refs[0].file_path)
    max_area = width * height * args.max_mask_area_frac if args.max_mask_area_frac > 0 else None
    records: List[Dict[str, Any]] = []

    try:
        for frame_idx in prompt_frames:
            LOG.info("Prompt %r on frame %d", prompt, frame_idx)
            predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": int(frame_idx),
                    "text": prompt,
                    "output_prob_thresh": args.output_prob_thresh,
                }
            )

        stream_request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": args.propagation_direction,
            "start_frame_index": min(prompt_frames) if prompt_frames else 0,
            "max_frame_num_to_track": len(frame_refs),
            "output_prob_thresh": args.output_prob_thresh,
        }
        for response in predictor.handle_stream_request(stream_request):
            frame_idx = response.get("frame_index")
            if frame_idx is None or frame_idx < 0 or frame_idx >= len(frame_refs):
                continue
            outputs = response.get("outputs", {})
            if not isinstance(outputs, Mapping):
                continue
            obj_ids = to_numpy(outputs.get("out_obj_ids"))
            boxes = to_numpy(outputs.get("out_boxes_xywh"))
            scores = to_numpy(outputs.get("out_probs"))
            masks = to_numpy(outputs.get("out_binary_masks"))
            if obj_ids.size == 0:
                continue
            frame_ref = frame_refs[int(frame_idx)]
            for local_idx, obj_id in enumerate(obj_ids.tolist()):
                if boxes.ndim >= 2 and local_idx < len(boxes):
                    bbox = denormalize_xywh(boxes[local_idx], width, height)
                elif masks.ndim >= 3 and local_idx < len(masks):
                    fallback = mask_to_bbox(masks[local_idx])
                    if fallback is None:
                        continue
                    bbox = fallback
                else:
                    continue
                area = bbox_area(bbox)
                if area < args.min_mask_area:
                    continue
                if max_area is not None and area > max_area:
                    continue
                score = float(scores[local_idx]) if scores.ndim >= 1 and local_idx < len(scores) else 1.0
                track_id = (
                    prompt_index * args.id_offset
                    + int(chunk_index) * args.chunk_id_offset
                    + int(obj_id)
                )
                records.append(
                    {
                        "video_id": frame_ref.video_id,
                        "image_id": frame_ref.image_id,
                        "category_id": 1,
                        "bbox_ltwh": np.asarray(bbox, dtype=np.float32),
                        "bbox_conf": score,
                        "track_id": int(track_id),
                        "track_bbox_ltwh": np.asarray(bbox, dtype=np.float32),
                        "track_bbox_conf": score,
                        "role_detection": "player",
                        "role_confidence": 1.0,
                        "role": "player",
                        "team_color_hint": team_color_hint,
                        "sam3_prompt": prompt,
                        "sam3_prompt_index": int(prompt_index),
                        "sam3_source_obj_id": int(obj_id),
                        "sam3_chunk_index": int(chunk_index),
                        "sam3_chunk_start_frame": int(chunk_start_frame),
                        "sam3_chunk_end_frame": int(chunk_end_frame),
                        "sam3_recondition_every": int(recondition_every),
                        "sam3_prompt_refresh_every": int(args.prompt_refresh_every or 0),
                    }
                )
    finally:
        try:
            predictor.handle_request(
                {"type": "close_session", "session_id": session_id, "run_gc_collect": True}
            )
        except Exception as exc:
            LOG.warning("Failed to close SAM3 session %s: %s", session_id, exc)
    return records


def dedupe_records(records: List[Dict[str, Any]], nms_iou: float) -> List[Dict[str, Any]]:
    if nms_iou <= 0:
        return records
    by_frame: Dict[Tuple[Any, int], List[Dict[str, Any]]] = {}
    for record in records:
        by_frame.setdefault((record["video_id"], int(record["image_id"])), []).append(record)
    kept: List[Dict[str, Any]] = []
    for frame_records in by_frame.values():
        ordered = sorted(
            frame_records,
            key=lambda r: (float(r.get("bbox_conf", 0.0)), bbox_area(r["bbox_ltwh"])),
            reverse=True,
        )
        frame_kept: List[Dict[str, Any]] = []
        for record in ordered:
            if all(bbox_iou(record["bbox_ltwh"], other["bbox_ltwh"]) < nms_iou for other in frame_kept):
                frame_kept.append(record)
        kept.extend(frame_kept)
    kept.sort(key=lambda r: (str(r["video_id"]), int(r["image_id"]), int(r["track_id"])))
    return kept


def stitch_chunk_tracks(records: List[Dict[str, Any]], stitch_iou: float) -> List[Dict[str, Any]]:
    if stitch_iou <= 0 or not records:
        return records

    parent: Dict[int, int] = {}

    def find(track_id: int) -> int:
        parent.setdefault(track_id, track_id)
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    def union(source: int, target: int) -> None:
        source_root = find(source)
        target_root = find(target)
        if source_root != target_root:
            parent[source_root] = target_root

    by_video_chunk: Dict[Tuple[Any, int], List[Dict[str, Any]]] = {}
    for record in records:
        video_id = record["video_id"]
        chunk_index = int(record.get("sam3_chunk_index", 0))
        by_video_chunk.setdefault((video_id, chunk_index), []).append(record)

    video_ids = sorted({video_id for video_id, _ in by_video_chunk.keys()}, key=str)
    for video_id in video_ids:
        chunk_indices = sorted(chunk for current_video, chunk in by_video_chunk if current_video == video_id)
        for previous_chunk, next_chunk in zip(chunk_indices, chunk_indices[1:]):
            previous_records = by_video_chunk[(video_id, previous_chunk)]
            next_records = by_video_chunk[(video_id, next_chunk)]
            previous_by_image = group_records_by_image(previous_records)
            next_by_image = group_records_by_image(next_records)
            common_image_ids = sorted(set(previous_by_image) & set(next_by_image))
            if not common_image_ids:
                continue

            pair_ious: Dict[Tuple[int, int], List[float]] = {}
            for image_id in common_image_ids:
                for previous in previous_by_image[image_id]:
                    previous_track_id = int(previous["track_id"])
                    for current in next_by_image[image_id]:
                        current_track_id = int(current["track_id"])
                        if previous_track_id == current_track_id:
                            continue
                        iou = bbox_iou(previous["bbox_ltwh"], current["bbox_ltwh"])
                        if iou > 0:
                            pair_ious.setdefault((previous_track_id, current_track_id), []).append(iou)

            candidates = sorted(
                (
                    (float(np.mean(values)), previous_track_id, current_track_id)
                    for (previous_track_id, current_track_id), values in pair_ious.items()
                    if float(np.mean(values)) >= stitch_iou
                ),
                reverse=True,
            )
            used_previous: set[int] = set()
            used_current: set[int] = set()
            for score, previous_track_id, current_track_id in candidates:
                if previous_track_id in used_previous or current_track_id in used_current:
                    continue
                union(current_track_id, previous_track_id)
                used_previous.add(previous_track_id)
                used_current.add(current_track_id)

    remapped = 0
    for record in records:
        old_track_id = int(record["track_id"])
        new_track_id = find(old_track_id)
        if new_track_id != old_track_id:
            record["track_id"] = new_track_id
            remapped += 1
    LOG.info("Chunk stitching remapped %d detections", remapped)
    return records


def group_records_by_image(records: Sequence[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_image: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        by_image.setdefault(int(record["image_id"]), []).append(record)
    return by_image


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=DETECTION_COLUMNS)
    df = pd.DataFrame(records)
    df = df.reset_index(drop=True)
    df["id"] = df.index.astype(int)
    for column in DETECTION_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df[DETECTION_COLUMNS]


def image_dataframe(frame_refs: Sequence[FrameRef]) -> pd.DataFrame:
    rows = [
        {
            "image_id": ref.image_id,
            "video_id": ref.video_id,
            "file_path": str(ref.file_path),
            "frame": ref.frame,
        }
        for ref in frame_refs
    ]
    df = pd.DataFrame(rows).set_index("image_id", drop=True)
    return df[IMAGE_COLUMNS]


def write_state(
    out_path: Path,
    detections_by_video: Mapping[str, pd.DataFrame],
    images_by_video: Mapping[str, pd.DataFrame],
    frame_refs_by_video: Mapping[str, Sequence[FrameRef]],
    args: argparse.Namespace,
    recondition_every: int,
) -> None:
    ensure_out_path(out_path, args.overwrite)
    summary = {
        "columns": {
            "detection": DETECTION_COLUMNS,
            "image": IMAGE_COLUMNS,
        },
        "sam3_builder": {
            "version": args.version,
            "mode": args.mode,
            "max_num_objects": args.max_num_objects,
            "chunk_frames": args.chunk_frames,
            "chunk_overlap": args.chunk_overlap,
            "chunk_id_offset": args.chunk_id_offset,
            "stitch_chunks": bool(args.stitch_chunks),
            "stitch_iou": args.stitch_iou,
            "recondition_every": recondition_every,
            "prompt_refresh_every": args.prompt_refresh_every,
            "offload_video_to_cpu": bool(args.offload_video_to_cpu),
            "offload_state_to_cpu": bool(args.offload_state_to_cpu),
            "prompts": list(args.prompts),
            "videos": list(frame_refs_by_video.keys()),
            "created_unix": time.time(),
        },
    }
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.writestr("summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
        for video, frame_refs in frame_refs_by_video.items():
            if not frame_refs:
                continue
            video_id = frame_refs[0].video_id
            det_df = detections_by_video.get(video, pd.DataFrame(columns=DETECTION_COLUMNS))
            img_df = images_by_video[video]
            with zf.open(f"{video_id}.pkl", "w", force_zip64=True) as fh:
                pickle.dump(det_df, fh, protocol=pickle.DEFAULT_PROTOCOL)
            with zf.open(f"{video_id}_image.pkl", "w", force_zip64=True) as fh:
                pickle.dump(img_df, fh, protocol=pickle.DEFAULT_PROTOCOL)
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest = {
        "out": str(out_path),
        "summary": summary["sam3_builder"],
        "videos": {
            video: {
                "video_id": refs[0].video_id,
                "num_frames": len(refs),
                "num_detections": int(len(detections_by_video.get(video, []))),
            }
            for video, refs in frame_refs_by_video.items()
        },
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    LOG.info("Wrote %s", out_path)
    LOG.info("Wrote %s", manifest_path)


def make_prompt_color_hints(args: argparse.Namespace) -> List[Optional[str]]:
    prompts = list(args.prompts)
    colors = args.prompt_team_colors
    if colors is None:
        return [None] * len(prompts)
    if len(colors) != len(prompts):
        raise ValueError("--prompt-team-colors must have the same length as --prompts")
    return [str(color) for color in colors]


def copy_frames_to_sam3_names(
    frame_refs: Sequence[FrameRef],
    tmp_root: Path,
    dirname: Optional[str] = None,
) -> Path:
    """SAM3 can read arbitrary image names, but zero-padded names avoid loader surprises."""
    out_dir = tmp_root / (dirname or frame_refs[0].video)
    out_dir.mkdir(parents=True, exist_ok=True)
    for local_index, ref in enumerate(frame_refs):
        suffix = ref.file_path.suffix.lower() or ".jpg"
        target = out_dir / f"{local_index:05d}{suffix}"
        if target.exists():
            continue
        try:
            os.link(ref.file_path, target)
        except OSError:
            shutil.copy2(ref.file_path, target)
    return out_dir


def run_builder(args: argparse.Namespace) -> None:
    root = split_root(args.dataset_root, args.split)
    videos = discover_videos(root, args.videos)
    metadata = load_metadata_state(args.metadata_state)
    video_id_map = load_json_mapping(args.video_id_map)
    frame_refs_by_video = prepare_frames(
        root,
        videos,
        metadata,
        video_id_map,
        args.nframes,
        args.preserve_metadata_file_path,
    )
    prompt_colors = make_prompt_color_hints(args)
    recondition_every = normalize_recondition_every(args)

    if args.mode == "periodic" and recondition_every <= 0 and args.prompt_refresh_every <= 0:
        raise ValueError("periodic mode requires --recondition-every > 0 or --prompt-refresh-every > 0")
    if args.dry_run:
        for video, refs in frame_refs_by_video.items():
            LOG.info("DRY RUN %s: %d frames, video_id=%d", video, len(refs), refs[0].video_id)
        return

    torch, predictor = build_predictor(args, recondition_every)
    detections_by_video: Dict[str, pd.DataFrame] = {}
    images_by_video: Dict[str, pd.DataFrame] = {
        video: image_dataframe(refs) for video, refs in frame_refs_by_video.items()
    }

    try:
        with tempfile.TemporaryDirectory(prefix="sam3_gsr_frames_") as tmp:
            tmp_root = Path(tmp)
            for video, refs in frame_refs_by_video.items():
                video_records: List[Dict[str, Any]] = []
                chunks = list(iter_frame_chunks(refs, args.chunk_frames, args.chunk_overlap))
                LOG.info(
                    "Running %s with %d prompts over %d chunks",
                    video,
                    len(args.prompts),
                    len(chunks),
                )
                for chunk_index, chunk_start, chunk_end, chunk_refs in chunks:
                    chunk_dirname = f"{video}_chunk_{chunk_index:04d}_{chunk_start:06d}_{chunk_end - 1:06d}"
                    frame_dir = copy_frames_to_sam3_names(chunk_refs, tmp_root, dirname=chunk_dirname)
                    prompt_frames = prompt_frames_for_chunk(
                        args,
                        chunk_start=chunk_start,
                        chunk_len=len(chunk_refs),
                        full_len=len(refs),
                    )
                    LOG.info(
                        "Running %s chunk %d/%d frames=%d..%d prompt_frames=%s",
                        video,
                        chunk_index + 1,
                        len(chunks),
                        chunk_start,
                        chunk_end - 1,
                        prompt_frames,
                    )
                    for prompt_index, (prompt, color) in enumerate(zip(args.prompts, prompt_colors)):
                        records = collect_prompt_records(
                            predictor=predictor,
                            frame_dir=frame_dir,
                            frame_refs=chunk_refs,
                            prompt=prompt,
                            prompt_index=prompt_index,
                            team_color_hint=color,
                            prompt_frames=prompt_frames,
                            chunk_index=chunk_index,
                            chunk_start_frame=chunk_start,
                            chunk_end_frame=chunk_end - 1,
                            args=args,
                            recondition_every=recondition_every,
                        )
                        LOG.info(
                            "%s chunk %d prompt %r produced %d detections",
                            video,
                            chunk_index,
                            prompt,
                            len(records),
                        )
                        video_records.extend(records)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    shutil.rmtree(frame_dir, ignore_errors=True)
                if args.stitch_chunks:
                    before_track_count = len({int(record["track_id"]) for record in video_records})
                    video_records = stitch_chunk_tracks(video_records, args.stitch_iou)
                    after_track_count = len({int(record["track_id"]) for record in video_records})
                    LOG.info(
                        "%s chunk stitch: %d -> %d track ids",
                        video,
                        before_track_count,
                        after_track_count,
                    )
                if not args.no_dedupe:
                    before = len(video_records)
                    video_records = dedupe_records(video_records, args.nms_iou)
                    LOG.info("%s dedupe: %d -> %d detections", video, before, len(video_records))
                detections_by_video[video] = records_to_dataframe(video_records)
    finally:
        close_predictor(predictor)

    write_state(
        out_path=args.out,
        detections_by_video=detections_by_video,
        images_by_video=images_by_video,
        frame_refs_by_video=frame_refs_by_video,
        args=args,
        recondition_every=recondition_every,
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    run_builder(args)


if __name__ == "__main__":
    main()
