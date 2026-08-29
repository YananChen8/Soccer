#!/usr/bin/env python3
"""Extract one video from a TrackLab sn-gamestate.pklz state.

This is useful when an older state used sequential or name-based video member
ids while the current SoccerNetGS loader expects ids such as 199. The tool finds
the requested video from image file paths, then writes a small normalized pklz
containing only <new_video_id>.pkl and <new_video_id>_image.pkl.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input TrackLab pklz.")
    parser.add_argument("--output", type=Path, help="Output normalized pklz.")
    parser.add_argument("--video", required=True, help="Video name, e.g. SNGS-199.")
    parser.add_argument(
        "--old-video-id",
        default=None,
        help="Known source member id. If omitted, infer from image file paths.",
    )
    parser.add_argument(
        "--new-video-id",
        default=None,
        help="Output member/video_id. Defaults to the numeric suffix from --video.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print detected members.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def infer_video_id(video: str) -> str:
    match = re.fullmatch(r"SNGS-(\d+)", video)
    if not match:
        raise ValueError(f"Cannot infer numeric video id from {video!r}; pass --new-video-id")
    return match.group(1)


def member_kind(member: str) -> tuple[Optional[str], Optional[str]]:
    if not member.endswith(".pkl"):
        return None, None
    stem = member[:-4]
    if stem.endswith("_image"):
        return "image", stem[: -len("_image")]
    return "detection", stem


def load_pickle(zf: zipfile.ZipFile, member: str) -> Any:
    with zf.open(member, "r") as fh:
        return pickle.load(fh)


def text_columns_contain(df: Any, text: str) -> bool:
    if not hasattr(df, "columns") or len(df) == 0:
        return False
    lowered = text.lower()
    candidate_columns = []
    if "file_path" in df.columns:
        candidate_columns.append("file_path")
    candidate_columns.extend(
        column for column in df.columns if column != "file_path" and getattr(df[column], "dtype", None) == object
    )
    for column in candidate_columns:
        values = df[column].dropna().astype(str)
        if values.str.lower().str.contains(lowered, regex=False).any():
            return True
    return False


def find_candidates(zf: zipfile.ZipFile, video: str, new_video_id: str) -> list[dict[str, Any]]:
    names = set(zf.namelist())
    image_members = []
    detection_members = []
    for member in sorted(names):
        kind, member_id = member_kind(member)
        if kind == "image":
            image_members.append((member_id, member))
        elif kind == "detection":
            detection_members.append((member_id, member))

    candidates: dict[str, dict[str, Any]] = {}

    for member_id, member in image_members:
        if member_id in {video, new_video_id}:
            candidates.setdefault(member_id, {})["reason"] = "member_id"
        try:
            image_df = load_pickle(zf, member)
        except Exception:
            continue
        if text_columns_contain(image_df, video):
            candidates.setdefault(member_id, {})["reason"] = "image_path"

    for member_id, member in detection_members:
        if member_id in {video, new_video_id}:
            candidates.setdefault(member_id, {})["reason"] = "member_id"
        if member_id in candidates:
            continue
        try:
            det_df = load_pickle(zf, member)
        except Exception:
            continue
        if hasattr(det_df, "columns") and "video_id" in det_df.columns:
            values = set(det_df["video_id"].dropna().astype(str).unique())
            if values & {video, new_video_id}:
                candidates.setdefault(member_id, {})["reason"] = "detection_video_id"

    result = []
    for member_id, info in sorted(candidates.items(), key=lambda item: item[0]):
        det_member = f"{member_id}.pkl"
        image_member = f"{member_id}_image.pkl"
        result.append(
            {
                "old_video_id": member_id,
                "reason": info.get("reason", "unknown"),
                "detection_member": det_member if det_member in names else None,
                "image_member": image_member if image_member in names else None,
            }
        )
    return result


def print_members(zf: zipfile.ZipFile) -> None:
    ids = []
    for member in zf.namelist():
        kind, member_id = member_kind(member)
        if kind == "detection":
            ids.append(member_id)
    print("Detection member ids:")
    print("  " + ", ".join(sorted(ids, key=str)) if ids else "  <none>")


def remap_df_video_id(df: Any, new_video_id: str) -> Any:
    if hasattr(df, "columns") and "video_id" in df.columns:
        df = df.copy()
        df.loc[:, "video_id"] = str(new_video_id)
    return df


def write_pickle(zf: zipfile.ZipFile, member: str, value: Any) -> None:
    with zf.open(member, "w", force_zip64=True) as fh:
        pickle.dump(value, fh, protocol=pickle.DEFAULT_PROTOCOL)


def copy_summary(zf_in: zipfile.ZipFile, zf_out: zipfile.ZipFile, old_video_id: str, new_video_id: str, video: str) -> None:
    if "summary.json" not in zf_in.namelist():
        return
    try:
        payload = json.loads(zf_in.read("summary.json").decode("utf-8"))
    except Exception:
        zf_out.writestr("summary.json", zf_in.read("summary.json"))
        return
    payload.setdefault("video_extract", []).append(
        {
            "video": video,
            "old_video_id": old_video_id,
            "new_video_id": new_video_id,
        }
    )
    zf_out.writestr("summary.json", json.dumps(payload, ensure_ascii=False, indent=2))


def select_old_video_id(args: argparse.Namespace, zf: zipfile.ZipFile, new_video_id: str) -> str:
    names = set(zf.namelist())
    if args.old_video_id is not None:
        old_video_id = str(args.old_video_id)
        if f"{old_video_id}.pkl" not in names:
            raise KeyError(f"{old_video_id}.pkl not found in {args.input}")
        return old_video_id

    candidates = find_candidates(zf, args.video, new_video_id)
    if not candidates:
        print(f"No member matched {args.video!r} or video_id {new_video_id!r}.")
        print_members(zf)
        raise SystemExit(3)
    if len(candidates) > 1:
        print(f"Multiple candidates matched {args.video!r}; rerun with --old-video-id:")
        for candidate in candidates:
            print(json.dumps(candidate, ensure_ascii=False))
        raise SystemExit(3)
    return str(candidates[0]["old_video_id"])


def main() -> None:
    args = parse_args()
    new_video_id = str(args.new_video_id or infer_video_id(args.video))
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    with zipfile.ZipFile(args.input, "r") as zf_in:
        old_video_id = select_old_video_id(args, zf_in, new_video_id)
        det_member = f"{old_video_id}.pkl"
        image_member = f"{old_video_id}_image.pkl"
        names = set(zf_in.namelist())
        if det_member not in names:
            raise KeyError(f"{det_member} not found in {args.input}")
        if image_member not in names:
            raise KeyError(f"{image_member} not found in {args.input}")

        print(f"Selected old_video_id={old_video_id} -> new_video_id={new_video_id}")
        print(f"  detection: {det_member} -> {new_video_id}.pkl")
        print(f"  image:     {image_member} -> {new_video_id}_image.pkl")

        if args.dry_run:
            return
        if args.output is None:
            raise ValueError("--output is required unless --dry-run is used")
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite")
        args.output.parent.mkdir(parents=True, exist_ok=True)

        output_path = args.output
        temp_dir = None
        if args.input.resolve() == args.output.resolve():
            temp_dir = Path(tempfile.mkdtemp(prefix="extract_tracker_state_video_"))
            output_path = temp_dir / args.output.name

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf_out:
            det_df = remap_df_video_id(load_pickle(zf_in, det_member), new_video_id)
            image_df = remap_df_video_id(load_pickle(zf_in, image_member), new_video_id)
            write_pickle(zf_out, f"{new_video_id}.pkl", det_df)
            write_pickle(zf_out, f"{new_video_id}_image.pkl", image_df)
            copy_summary(zf_in, zf_out, old_video_id, new_video_id, args.video)

        if temp_dir is not None:
            shutil.move(str(output_path), str(args.output))
            shutil.rmtree(str(temp_dir), ignore_errors=True)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
