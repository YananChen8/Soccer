#!/usr/bin/env python3
"""Evaluate image-to-pitch Homographies in a TrackLab state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refine_nbjw_with_b2p import (  # noqa: E402
    json_default,
    load_homographies_for_eval,
    normalize_video,
    split_root,
)
from sn_gamestate.structured_calibration.metrics import homography_accuracy_eval  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-pklz", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Pixel gate for line comparison. Use 5.0 for JaC@5-style reporting.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = [normalize_video(video) for video in args.videos]
    homographies = load_homographies_for_eval(args.state_pklz, videos)
    result: dict[str, Any] = {
        "state_pklz": str(args.state_pklz),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "videos": videos,
        "threshold_px": float(args.threshold),
        "homography_eval": homography_accuracy_eval(
            homographies,
            split_root(args.dataset_root, args.split),
            videos,
            threshold=args.threshold,
            nproc=args.nproc,
            stride=args.stride,
        ),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, default=json_default)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
