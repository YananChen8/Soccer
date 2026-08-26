#!/usr/bin/env python3
"""Remap one video id inside a TrackLab ``sn-gamestate.pklz`` state file."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input pklz state.")
    parser.add_argument("--output", type=Path, required=True, help="Output pklz state.")
    parser.add_argument("--old-video-id", required=True)
    parser.add_argument("--new-video-id", default=None)
    parser.add_argument("--video", default=None, help="Optional name such as SNGS-021; implies --new-video-id 021.")
    parser.add_argument(
        "--add-image-eval-bbox-pitch",
        action="store_true",
        help=(
            "Add a non-empty bbox_pitch placeholder from bbox_ltwh bottom-center. "
            "Use this only for image-space SoccerNetGS eval exports."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def infer_video_id(video: Optional[str]) -> Optional[str]:
    if not video:
        return None
    match = re.fullmatch(r"SNGS-(\d+)", video)
    return match.group(1) if match else None


def resolve_new_video_id(args: argparse.Namespace) -> str:
    if args.new_video_id is not None:
        return str(args.new_video_id)
    inferred = infer_video_id(args.video)
    if inferred is not None:
        return inferred
    raise ValueError("Pass --new-video-id or --video SNGS-XXX")


def valid_bbox_pitch(value: Any) -> bool:
    return isinstance(value, dict) and "x_bottom_middle" in value and "y_bottom_middle" in value


def bbox_to_image_eval_bbox_pitch(box: Sequence[float]) -> dict[str, float]:
    left, top, width, height = [float(value) for value in box[:4]]
    return {
        "x_bottom_middle": left + width / 2.0,
        "y_bottom_middle": top + height,
    }


def add_image_eval_bbox_pitch(df: Any) -> Any:
    if "bbox_ltwh" not in df.columns:
        raise KeyError("Cannot add bbox_pitch placeholder because bbox_ltwh is missing")
    df = df.copy()
    existing = df["bbox_pitch"] if "bbox_pitch" in df.columns else [None] * len(df)
    df["bbox_pitch"] = [
        current if valid_bbox_pitch(current) else bbox_to_image_eval_bbox_pitch(bbox)
        for current, bbox in zip(existing, df["bbox_ltwh"])
    ]
    return df


def add_detection_column_to_summary(payload: dict[str, Any], column: str) -> None:
    columns = payload.get("columns")
    if isinstance(columns, dict):
        detection_columns = columns.setdefault("detection", [])
    elif isinstance(columns, list):
        detection_columns = columns
    else:
        payload["columns"] = {"detection": [column], "image": []}
        return
    if column not in detection_columns:
        detection_columns.append(column)


def remap_dataframe_member(
    zf_in: zipfile.ZipFile,
    zf_out: zipfile.ZipFile,
    member: str,
    out_member: str,
    new_video_id: str,
    add_bbox_pitch: bool = False,
) -> None:
    with zf_in.open(member, "r") as fh:
        df = pickle.load(fh)
    if "video_id" in df.columns:
        df.loc[:, "video_id"] = str(new_video_id)
    if add_bbox_pitch:
        df = add_image_eval_bbox_pitch(df)
    with zf_out.open(out_member, "w", force_zip64=True) as fh:
        pickle.dump(df, fh, protocol=pickle.DEFAULT_PROTOCOL)


def remap_member_name(member: str, old_video_id: str, new_video_id: str) -> str:
    if member == f"{old_video_id}.pkl":
        return f"{new_video_id}.pkl"
    if member == f"{old_video_id}_image.pkl":
        return f"{new_video_id}_image.pkl"
    return member


def main() -> None:
    args = parse_args()
    new_video_id = resolve_new_video_id(args)
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    output_path = args.output
    temp_path = None
    if args.input.resolve() == args.output.resolve():
        temp_dir = Path(tempfile.mkdtemp(prefix="remap_tracker_state_"))
        temp_path = temp_dir / args.output.name
        output_path = temp_path

    remapped_members = []
    with zipfile.ZipFile(args.input, "r") as zf_in:
        names = zf_in.namelist()
        old_det = f"{args.old_video_id}.pkl"
        old_img = f"{args.old_video_id}_image.pkl"
        if old_det not in names and old_img not in names:
            raise KeyError(f"Neither {old_det} nor {old_img} exists in {args.input}")
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf_out:
            for member in names:
                out_member = remap_member_name(member, args.old_video_id, new_video_id)
                if member in {old_det, old_img}:
                    remap_dataframe_member(
                        zf_in,
                        zf_out,
                        member,
                        out_member,
                        new_video_id,
                        add_bbox_pitch=args.add_image_eval_bbox_pitch and member == old_det,
                    )
                    remapped_members.append((member, out_member))
                elif member == "summary.json":
                    payload = json.loads(zf_in.read(member).decode("utf-8"))
                    payload.setdefault("video_id_remap", []).append(
                        {"old_video_id": args.old_video_id, "new_video_id": new_video_id}
                    )
                    if args.add_image_eval_bbox_pitch:
                        add_detection_column_to_summary(payload, "bbox_pitch")
                        payload.setdefault("state_patches", []).append(
                            {
                                "name": "image_eval_bbox_pitch_placeholder",
                                "source": "bbox_ltwh bottom-center in image coordinates",
                            }
                        )
                    zf_out.writestr(member, json.dumps(payload, ensure_ascii=False, indent=2))
                else:
                    zf_out.writestr(out_member, zf_in.read(member))

    if temp_path is not None:
        shutil.move(str(output_path), str(args.output))
        shutil.rmtree(str(temp_path.parent), ignore_errors=True)

    print(f"Remapped {args.input} -> {args.output}")
    for old, new in remapped_members:
        print(f"  {old} -> {new}")


if __name__ == "__main__":
    main()
