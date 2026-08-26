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
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input pklz state.")
    parser.add_argument("--output", type=Path, required=True, help="Output pklz state.")
    parser.add_argument("--old-video-id", type=int, required=True)
    parser.add_argument("--new-video-id", type=int, default=None)
    parser.add_argument("--video", default=None, help="Optional name such as SNGS-021; implies --new-video-id 21.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def infer_video_id(video: Optional[str]) -> Optional[int]:
    if not video:
        return None
    match = re.fullmatch(r"SNGS-(\d+)", video)
    return int(match.group(1)) if match else None


def resolve_new_video_id(args: argparse.Namespace) -> int:
    if args.new_video_id is not None:
        return int(args.new_video_id)
    inferred = infer_video_id(args.video)
    if inferred is not None:
        return inferred
    raise ValueError("Pass --new-video-id or --video SNGS-XXX")


def remap_dataframe_member(
    zf_in: zipfile.ZipFile,
    zf_out: zipfile.ZipFile,
    member: str,
    out_member: str,
    new_video_id: int,
) -> None:
    with zf_in.open(member, "r") as fh:
        df = pickle.load(fh)
    if "video_id" in df.columns:
        df.loc[:, "video_id"] = int(new_video_id)
    with zf_out.open(out_member, "w", force_zip64=True) as fh:
        pickle.dump(df, fh, protocol=pickle.DEFAULT_PROTOCOL)


def remap_member_name(member: str, old_video_id: int, new_video_id: int) -> str:
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
                    remap_dataframe_member(zf_in, zf_out, member, out_member, new_video_id)
                    remapped_members.append((member, out_member))
                elif member == "summary.json":
                    payload = json.loads(zf_in.read(member).decode("utf-8"))
                    payload.setdefault("video_id_remap", []).append(
                        {"old_video_id": args.old_video_id, "new_video_id": new_video_id}
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
