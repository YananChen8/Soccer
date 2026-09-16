#!/usr/bin/env python3
"""Merge TrackLab ``sn-gamestate.pklz`` state files.

TrackLab stores one detection pickle and one image pickle per video inside the
zip file, for example ``0.pkl`` and ``0_image.pkl``. This helper merges several
single-video or split-video states into one state that can be loaded by the
standard TrackLab evaluator.
"""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def member_kind(member: str) -> tuple[Optional[str], Optional[str]]:
    if not member.endswith(".pkl"):
        return None, None
    stem = member[:-4]
    if stem.endswith("_image"):
        return "image", stem[: -len("_image")]
    return "detection", stem


def read_pickle(zf: zipfile.ZipFile, member: str) -> Any:
    with zf.open(member, "r") as fp:
        return pickle.load(fp)


def write_pickle(zf: zipfile.ZipFile, member: str, value: Any) -> None:
    with zf.open(member, "w", force_zip64=True) as fp:
        pickle.dump(value, fp, protocol=pickle.DEFAULT_PROTOCOL)


def columns_of(df: Any) -> list[str]:
    if hasattr(df, "columns"):
        return [str(column) for column in df.columns]
    return []


def reindex_df(df: Any, columns: list[str]) -> Any:
    if hasattr(df, "reindex") and hasattr(df, "columns"):
        return df.reindex(columns=columns)
    return df


def stable_union(groups: list[list[str]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return result


def video_id_sort_key(value: str) -> tuple[int, Any]:
    text = str(value)
    if text.isdigit():
        return 0, int(text)
    return 1, text


def main() -> None:
    args = parse_args()
    for path in args.inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")

    detections: dict[str, Any] = {}
    images: dict[str, Any] = {}
    det_column_groups: list[list[str]] = []
    image_column_groups: list[list[str]] = []
    sources: list[dict[str, Any]] = []

    for path in args.inputs:
        copied_members: list[str] = []
        with zipfile.ZipFile(path, "r") as zf:
            for member in sorted(zf.namelist()):
                kind, video_id = member_kind(member)
                if kind is None or video_id is None:
                    continue
                if kind == "detection":
                    if video_id in detections:
                        raise ValueError(f"Duplicate detection member id {video_id!r} from {path}")
                    df = read_pickle(zf, member)
                    detections[video_id] = df
                    det_column_groups.append(columns_of(df))
                elif kind == "image":
                    if video_id in images:
                        raise ValueError(f"Duplicate image member id {video_id!r} from {path}")
                    df = read_pickle(zf, member)
                    images[video_id] = df
                    image_column_groups.append(columns_of(df))
                copied_members.append(member)
        sources.append({"path": str(path), "members": copied_members})

    missing_images = sorted(set(detections) - set(images))
    missing_detections = sorted(set(images) - set(detections))
    if missing_images:
        raise ValueError(f"Missing image members for video ids: {missing_images}")
    if missing_detections:
        raise ValueError(f"Missing detection members for video ids: {missing_detections}")

    det_columns = stable_union(det_column_groups)
    image_columns = stable_union(image_column_groups)
    if "image_id" in det_columns:
        det_columns = ["image_id"] + [column for column in det_columns if column != "image_id"]
    if "video_id" in det_columns:
        det_columns = ["video_id"] + [column for column in det_columns if column != "video_id"]
    if "video_id" in image_columns:
        image_columns = ["video_id"] + [column for column in image_columns if column != "video_id"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=args.output.name + ".",
        suffix=".tmp",
        dir=args.output.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            summary = {
                "columns": {
                    "detection": det_columns,
                    "image": image_columns,
                },
                "merged_tracker_state": {
                    "source_states": sources,
                    "video_ids": sorted(detections, key=video_id_sort_key),
                },
            }
            zf.writestr("summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
            for video_id in sorted(detections, key=video_id_sort_key):
                write_pickle(zf, f"{video_id}.pkl", reindex_df(detections[video_id], det_columns))
                write_pickle(zf, f"{video_id}_image.pkl", reindex_df(images[video_id], image_columns))

        tmp_path.replace(args.output)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    print(f"Wrote {args.output}")
    print(f"Merged {len(detections)} videos from {len(args.inputs)} state files.")


if __name__ == "__main__":
    main()
