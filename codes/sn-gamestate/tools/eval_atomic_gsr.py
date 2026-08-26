#!/usr/bin/env python3
"""Standalone atomic metrics for SoccerNet Game State Reconstruction.

This evaluator is meant for ablations where each task is measured separately:

* image-space DetA / AssA from HOTA, using bbox IoU only;
* optional pitch-space LocA from HOTA, using bbox_pitch only when it is real;
* track-level Role Macro-F1, Team Accuracy, and Jersey Exact Accuracy.

It deliberately avoids TrackLab's SoccerNetGS export wrapper, so a partial state
from SAM3 can be evaluated without requiring role/team/jersey/pitch fields that
belong to later pipeline stages.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from eval_atomic_attributes import (
    Detection,
    dedupe_detections,
    evaluate_attributes,
    iou_ltwh,
    json_ready,
    linear_assignment_max,
    load_gt_dataset,
    load_pred_json_dir,
    load_state_detections,
    natural_key,
    normalize_video,
    parse_state_video_map,
    safe_div,
)


HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)
PITCH_MATCH_FLOOR = 0.05


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
    parser.add_argument("--distance-tol", type=float, default=5.0)
    parser.add_argument("--min-track-matches", type=int, default=1)
    parser.add_argument("--include-missing-gt-jersey", action="store_true")
    parser.add_argument("--keep-ball", action="store_true")
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument(
        "--skip-attributes",
        action="store_true",
        help="Only compute image/pitch HOTA submetrics.",
    )
    parser.add_argument(
        "--force-pitch",
        action="store_true",
        help="Compute pitch LocA even if bbox_pitch values look like image-coordinate placeholders.",
    )
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def hota_empty_result() -> Dict[str, np.ndarray]:
    return {
        "HOTA_TP": np.zeros_like(HOTA_ALPHAS, dtype=float),
        "HOTA_FN": np.zeros_like(HOTA_ALPHAS, dtype=float),
        "HOTA_FP": np.zeros_like(HOTA_ALPHAS, dtype=float),
        "HOTA": np.zeros_like(HOTA_ALPHAS, dtype=float),
        "DetA": np.zeros_like(HOTA_ALPHAS, dtype=float),
        "AssA": np.zeros_like(HOTA_ALPHAS, dtype=float),
        "DetRe": np.zeros_like(HOTA_ALPHAS, dtype=float),
        "DetPr": np.zeros_like(HOTA_ALPHAS, dtype=float),
        "AssRe": np.zeros_like(HOTA_ALPHAS, dtype=float),
        "AssPr": np.zeros_like(HOTA_ALPHAS, dtype=float),
        "LocA": np.zeros_like(HOTA_ALPHAS, dtype=float),
    }


def finalize_hota_result(result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    result["DetRe"] = result["HOTA_TP"] / np.maximum(1.0, result["HOTA_TP"] + result["HOTA_FN"])
    result["DetPr"] = result["HOTA_TP"] / np.maximum(1.0, result["HOTA_TP"] + result["HOTA_FP"])
    result["DetA"] = result["HOTA_TP"] / np.maximum(
        1.0,
        result["HOTA_TP"] + result["HOTA_FN"] + result["HOTA_FP"],
    )
    result["HOTA"] = np.sqrt(result["DetA"] * result["AssA"])
    return result


def gaussian_sigma(distance_tol: float) -> float:
    return float(distance_tol) / math.sqrt(-2.0 * math.log(PITCH_MATCH_FLOOR))


def pitch_similarity(gt: Detection, pred: Detection, sigma: float) -> float:
    if gt.pitch_xy is None or pred.pitch_xy is None:
        return 0.0
    dx = float(gt.pitch_xy[0] - pred.pitch_xy[0])
    dy = float(gt.pitch_xy[1] - pred.pitch_xy[1])
    return math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))


def similarity(gt: Detection, pred: Detection, space: str, sigma: float) -> float:
    if space == "pitch":
        return pitch_similarity(gt, pred, sigma)
    return iou_ltwh(gt.bbox_ltwh, pred.bbox_ltwh)


def group_by_frame(detections: Iterable[Detection]) -> Dict[int, list[Detection]]:
    grouped: Dict[int, list[Detection]] = defaultdict(list)
    for det in detections:
        grouped[int(det.frame)].append(det)
    return grouped


def track_index(detections: Sequence[Detection]) -> Dict[str, int]:
    tracks = sorted({det.track_id for det in detections}, key=natural_key)
    return {track: index for index, track in enumerate(tracks)}


def sequence_hota(
    gt_detections: Sequence[Detection],
    pred_detections: Sequence[Detection],
    space: str,
    distance_tol: float,
) -> Dict[str, np.ndarray]:
    result = hota_empty_result()
    num_gt_dets = len(gt_detections)
    num_pred_dets = len(pred_detections)
    if num_pred_dets == 0:
        result["HOTA_FN"] = num_gt_dets * np.ones_like(HOTA_ALPHAS, dtype=float)
        result["LocA"] = np.ones_like(HOTA_ALPHAS, dtype=float)
        return finalize_hota_result(result)
    if num_gt_dets == 0:
        result["HOTA_FP"] = num_pred_dets * np.ones_like(HOTA_ALPHAS, dtype=float)
        result["LocA"] = np.ones_like(HOTA_ALPHAS, dtype=float)
        return finalize_hota_result(result)

    gt_track_to_idx = track_index(gt_detections)
    pred_track_to_idx = track_index(pred_detections)
    gt_id_count = np.zeros((len(gt_track_to_idx), 1), dtype=float)
    pred_id_count = np.zeros((1, len(pred_track_to_idx)), dtype=float)
    potential_matches_count = np.zeros((len(gt_track_to_idx), len(pred_track_to_idx)), dtype=float)

    gt_by_frame = group_by_frame(gt_detections)
    pred_by_frame = group_by_frame(pred_detections)
    frames = sorted(set(gt_by_frame) | set(pred_by_frame))
    sigma = gaussian_sigma(distance_tol)
    timestep_data = []

    for frame in frames:
        gt_frame = gt_by_frame.get(frame, [])
        pred_frame = pred_by_frame.get(frame, [])
        gt_ids = np.asarray([gt_track_to_idx[det.track_id] for det in gt_frame], dtype=int)
        pred_ids = np.asarray([pred_track_to_idx[det.track_id] for det in pred_frame], dtype=int)
        sim = np.asarray(
            [[similarity(gt, pred, space, sigma) for pred in pred_frame] for gt in gt_frame],
            dtype=float,
        )
        timestep_data.append((gt_ids, pred_ids, sim))
        if len(gt_ids):
            gt_id_count[gt_ids] += 1
        if len(pred_ids):
            pred_id_count[0, pred_ids] += 1
        if sim.size:
            denom = sim.sum(axis=0)[np.newaxis, :] + sim.sum(axis=1)[:, np.newaxis] - sim
            sim_iou = np.zeros_like(sim)
            mask = denom > np.finfo(float).eps
            sim_iou[mask] = sim[mask] / denom[mask]
            potential_matches_count[gt_ids[:, np.newaxis], pred_ids[np.newaxis, :]] += sim_iou

    denom = gt_id_count + pred_id_count - potential_matches_count
    global_alignment = np.zeros_like(potential_matches_count)
    mask = denom > np.finfo(float).eps
    global_alignment[mask] = potential_matches_count[mask] / denom[mask]
    matches_counts = [np.zeros_like(potential_matches_count) for _ in HOTA_ALPHAS]

    for gt_ids, pred_ids, sim in timestep_data:
        if len(gt_ids) == 0:
            result["HOTA_FP"] += len(pred_ids)
            continue
        if len(pred_ids) == 0:
            result["HOTA_FN"] += len(gt_ids)
            continue

        score_mat = global_alignment[gt_ids[:, np.newaxis], pred_ids[np.newaxis, :]] * sim
        assignments = linear_assignment_max(score_mat.tolist())
        if not assignments:
            result["HOTA_FN"] += len(gt_ids)
            result["HOTA_FP"] += len(pred_ids)
            continue
        rows = np.asarray([row for row, _ in assignments], dtype=int)
        cols = np.asarray([col for _, col in assignments], dtype=int)
        matched_sim = sim[rows, cols]

        for alpha_index, alpha in enumerate(HOTA_ALPHAS):
            actually_matched = matched_sim >= alpha - np.finfo(float).eps
            alpha_rows = rows[actually_matched]
            alpha_cols = cols[actually_matched]
            num_matches = len(alpha_rows)
            result["HOTA_TP"][alpha_index] += num_matches
            result["HOTA_FN"][alpha_index] += len(gt_ids) - num_matches
            result["HOTA_FP"][alpha_index] += len(pred_ids) - num_matches
            if num_matches:
                result["LocA"][alpha_index] += float(np.sum(sim[alpha_rows, alpha_cols]))
                matches_counts[alpha_index][gt_ids[alpha_rows], pred_ids[alpha_cols]] += 1

    for alpha_index, _ in enumerate(HOTA_ALPHAS):
        matches_count = matches_counts[alpha_index]
        ass_a = matches_count / np.maximum(1.0, gt_id_count + pred_id_count - matches_count)
        ass_re = matches_count / np.maximum(1.0, gt_id_count)
        ass_pr = matches_count / np.maximum(1.0, pred_id_count)
        result["AssA"][alpha_index] = np.sum(matches_count * ass_a) / np.maximum(
            1.0,
            result["HOTA_TP"][alpha_index],
        )
        result["AssRe"][alpha_index] = np.sum(matches_count * ass_re) / np.maximum(
            1.0,
            result["HOTA_TP"][alpha_index],
        )
        result["AssPr"][alpha_index] = np.sum(matches_count * ass_pr) / np.maximum(
            1.0,
            result["HOTA_TP"][alpha_index],
        )

    result["LocA"] = np.maximum(1e-10, result["LocA"]) / np.maximum(1e-10, result["HOTA_TP"])
    return finalize_hota_result(result)


def combine_hota(results: Mapping[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    combined = hota_empty_result()
    if not results:
        return finalize_hota_result(combined)
    for field in ("HOTA_TP", "HOTA_FN", "HOTA_FP"):
        combined[field] = sum(result[field] for result in results.values())
    tp = np.maximum(1e-10, combined["HOTA_TP"])
    for field in ("AssA", "AssRe", "AssPr", "LocA"):
        weighted = sum(result[field] * result["HOTA_TP"] for result in results.values())
        combined[field] = np.maximum(1e-10, weighted) / tp
    return finalize_hota_result(combined)


def summarize_hota(result: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    alpha_50 = int(np.argmin(np.abs(HOTA_ALPHAS - 0.5)))
    summary: Dict[str, Any] = {
        "HOTA": float(np.mean(result["HOTA"])),
        "DetA": float(np.mean(result["DetA"])),
        "AssA": float(np.mean(result["AssA"])),
        "LocA": float(np.mean(result["LocA"])),
        "DetRe": float(np.mean(result["DetRe"])),
        "DetPr": float(np.mean(result["DetPr"])),
        "AssRe": float(np.mean(result["AssRe"])),
        "AssPr": float(np.mean(result["AssPr"])),
        "HOTA_TP": int(np.mean(result["HOTA_TP"])),
        "HOTA_FN": int(np.mean(result["HOTA_FN"])),
        "HOTA_FP": int(np.mean(result["HOTA_FP"])),
        "HOTA@0.50": float(result["HOTA"][alpha_50]),
        "DetA@0.50": float(result["DetA"][alpha_50]),
        "AssA@0.50": float(result["AssA"][alpha_50]),
        "LocA@0.50": float(result["LocA"][alpha_50]),
    }
    return summary


def hota_to_json(result: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    return {
        "alphas": [float(alpha) for alpha in HOTA_ALPHAS],
        "summary": summarize_hota(result),
        "arrays": {key: [float(value) for value in values] for key, values in result.items()},
    }


def duplicate_stats(detections: Iterable[Detection]) -> Dict[str, Any]:
    counts = Counter((det.video, det.frame, det.track_id) for det in detections)
    duplicates = [(key, count) for key, count in counts.items() if count > 1]
    return {
        "duplicate_frame_track_rows": int(sum(count - 1 for _, count in duplicates)),
        "duplicate_frame_track_keys": int(len(duplicates)),
        "examples": [
            {"video": key[0], "frame": key[1], "track_id": key[2], "count": count}
            for key, count in duplicates[:10]
        ],
    }


def pitch_counts(detections: Iterable[Detection]) -> Dict[str, Any]:
    points = [det.pitch_xy for det in detections if det.pitch_xy is not None]
    if not points:
        return {
            "with_pitch": 0,
            "max_abs_x": None,
            "max_abs_y": None,
            "looks_like_pitch": False,
        }
    max_abs_x = max(abs(float(x)) for x, _ in points)
    max_abs_y = max(abs(float(y)) for _, y in points)
    return {
        "with_pitch": len(points),
        "max_abs_x": max_abs_x,
        "max_abs_y": max_abs_y,
        "looks_like_pitch": max_abs_x <= 120.0 and max_abs_y <= 90.0,
    }


def pitch_eval_ready(
    gt_by_video: Mapping[str, Sequence[Detection]],
    pred_by_video: Mapping[str, Sequence[Detection]],
    source_metadata: Mapping[str, Any],
    force: bool,
) -> Tuple[bool, str]:
    if not force and state_marks_placeholder_pitch(source_metadata):
        return (
            False,
            "state metadata marks bbox_pitch as an image-eval placeholder; use a calibrated state for LocA.",
        )
    all_gt = [det for detections in gt_by_video.values() for det in detections]
    all_pred = [det for detections in pred_by_video.values() for det in detections]
    gt_pitch = pitch_counts(all_gt)
    pred_pitch = pitch_counts(all_pred)
    if gt_pitch["with_pitch"] == 0:
        return False, "GT has no bbox_pitch."
    if pred_pitch["with_pitch"] == 0:
        return False, "predictions have no bbox_pitch."
    if not force and not pred_pitch["looks_like_pitch"]:
        return (
            False,
            "prediction bbox_pitch looks like image-coordinate placeholders; "
            "use a calibrated state or pass --force-pitch to override.",
        )
    return True, "ok"


def read_state_metadata(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        with zipfile.ZipFile(path, "r") as zf:
            if "summary.json" not in zf.namelist():
                return {}
            payload = json.loads(zf.read("summary.json").decode("utf-8"))
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def state_marks_placeholder_pitch(metadata: Mapping[str, Any]) -> bool:
    sam3_builder = metadata.get("sam3_builder")
    if isinstance(sam3_builder, Mapping):
        marker = str(sam3_builder.get("bbox_pitch", "")).lower()
        if "placeholder" in marker:
            return True
    patches = metadata.get("state_patches", [])
    if isinstance(patches, Sequence) and not isinstance(patches, (str, bytes)):
        for patch in patches:
            if isinstance(patch, Mapping):
                name = str(patch.get("name", "")).lower()
                source = str(patch.get("source", "")).lower()
                if "bbox_pitch" in name and "placeholder" in name:
                    return True
                if "bottom-center in image coordinates" in source:
                    return True
    return False


def evaluate_hota(
    gt_by_video: Mapping[str, Sequence[Detection]],
    pred_by_video: Mapping[str, Sequence[Detection]],
    space: str,
    distance_tol: float,
) -> Dict[str, Any]:
    videos = sorted(set(gt_by_video) & set(pred_by_video))
    per_video_arrays = {
        video: sequence_hota(gt_by_video[video], pred_by_video[video], space, distance_tol)
        for video in videos
    }
    combined = combine_hota(per_video_arrays)
    return {
        "summary": summarize_hota(combined),
        "combined": hota_to_json(combined),
        "per_video": {video: hota_to_json(result) for video, result in per_video_arrays.items()},
    }


def percent(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.3f}"


def print_report(result: Mapping[str, Any]) -> None:
    image = result["image_hota"]["summary"]
    print("Atomic GSR metrics")
    print(f"  Image DetA:                 {percent(image['DetA'])}")
    print(f"  Image AssA:                 {percent(image['AssA'])}")
    print(f"  Image HOTA:                 {percent(image['HOTA'])}")
    print(f"  Image LocA/IoU:             {percent(image['LocA'])}")

    pitch = result.get("pitch_hota")
    if pitch is not None:
        print(f"  Pitch LocA:                 {percent(pitch['summary']['LocA'])}")
    else:
        print(f"  Pitch LocA:                 skipped ({result['pitch_skip_reason']})")

    attrs = result.get("attributes")
    if attrs is not None:
        summary = attrs["summary"]
        print(f"  RoleMacroF1:                {percent(summary['RoleMacroF1'])}")
        print(f"  TeamTrackAccuracy:          {percent(summary['TeamTrackAccuracy'])}")
        print(f"  JerseyTrackExactAccuracy:   {percent(summary['JerseyTrackExactAccuracy'])}")
        print(f"  matched_tracks:             {summary['matched_tracks']}")
    print(f"  videos:                     {result['diagnostics']['videos']}")
    print(f"  gt_detections:              {result['diagnostics']['gt_detections']}")
    print(f"  pred_detections:            {result['diagnostics']['pred_detections']}")
    duplicate_rows = result["diagnostics"]["pred_duplicates_before_dedupe"]["duplicate_frame_track_rows"]
    if duplicate_rows:
        print(f"  pred duplicate rows fixed:  {duplicate_rows}")


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
        source_metadata = read_state_metadata(args.state_pklz)
        pred_by_video = load_state_detections(
            args.state_pklz,
            videos,
            args.bbox_format,
            ignore_ball,
            parse_state_video_map(args.state_video_map),
        )
    else:
        source_metadata = {}
        pred_by_video = load_pred_json_dir(args.pred_dir, videos, args.bbox_format, ignore_ball)

    missing_pred = sorted(set(gt_by_video) - set(pred_by_video))
    if missing_pred:
        raise FileNotFoundError(f"Missing predictions for: {', '.join(missing_pred)}")

    all_gt_before = [det for detections in gt_by_video.values() for det in detections]
    all_pred_before = [det for detections in pred_by_video.values() for det in detections]
    gt_duplicates = duplicate_stats(all_gt_before)
    pred_duplicates = duplicate_stats(all_pred_before)

    if not args.no_dedupe:
        gt_by_video = {
            video: dedupe_detections(detections)
            for video, detections in gt_by_video.items()
        }
        pred_by_video = {
            video: dedupe_detections(detections)
            for video, detections in pred_by_video.items()
        }

    result: Dict[str, Any] = {
        "config": {
            "dataset_root": str(args.dataset_root),
            "split": args.split,
            "videos": videos or sorted(gt_by_video),
            "source": str(args.state_pklz or args.pred_dir),
            "bbox_format": args.bbox_format,
            "distance_tol": args.distance_tol,
            "dedupe": not args.no_dedupe,
            "ignore_ball": ignore_ball,
        },
        "diagnostics": {
            "videos": len(set(gt_by_video) & set(pred_by_video)),
            "gt_detections": sum(len(detections) for detections in gt_by_video.values()),
            "pred_detections": sum(len(detections) for detections in pred_by_video.values()),
            "gt_duplicates_before_dedupe": gt_duplicates,
            "pred_duplicates_before_dedupe": pred_duplicates,
            "gt_pitch": pitch_counts(det for detections in gt_by_video.values() for det in detections),
            "pred_pitch": pitch_counts(det for detections in pred_by_video.values() for det in detections),
            "state_marks_placeholder_pitch": state_marks_placeholder_pitch(source_metadata),
        },
    }

    result["image_hota"] = evaluate_hota(gt_by_video, pred_by_video, "image", args.distance_tol)

    pitch_ready, pitch_reason = pitch_eval_ready(gt_by_video, pred_by_video, source_metadata, args.force_pitch)
    if pitch_ready:
        result["pitch_hota"] = evaluate_hota(gt_by_video, pred_by_video, "pitch", args.distance_tol)
        result["pitch_skip_reason"] = None
    else:
        result["pitch_hota"] = None
        result["pitch_skip_reason"] = pitch_reason

    if not args.skip_attributes:
        result["attributes"] = evaluate_attributes(
            gt_by_video,
            pred_by_video,
            iou_threshold=0.5,
            min_track_matches=args.min_track_matches,
            include_missing_gt_jersey=args.include_missing_gt_jersey,
        )

    print_report(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(json_ready(result), f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
