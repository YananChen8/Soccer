#!/usr/bin/env python3
"""Export K=5 per-outlier cases for frames with many RANSAC outliers."""
import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cached_full_test_round2 import (
    DATA_ROOT,
    complete_keypoints,
    coords_to_dict,
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    ransac_inlier_keys,
)
from sn_gamestate.structured_calibration.metrics import load_gt_lines_for_video


def decode_npz(path):
    d = np.load(path)
    kp = torch.from_numpy(d["kp_hm"].astype(np.float32)).unsqueeze(0)
    line = torch.from_numpy(d["line_hm"].astype(np.float32)).unsqueeze(0)
    kc = get_keypoints_from_heatmap_batch_maxpool(kp[:, :-1])
    lc = get_keypoints_from_heatmap_batch_maxpool_l(line[:, :-1])
    pred = complete_keypoints(
        coords_to_dict(kc, threshold=0.1449),
        coords_to_dict(lc, threshold=0.2983),
        w=960,
        h=540,
        normalize=True,
    )[0]
    return int(d["frame"]), pred


def pack(pt):
    return {"x": float(pt["x"]), "y": float(pt["y"]), "p": float(pt.get("p", 1.0))}


def load_video(cache_root, vid, stride):
    files = sorted(glob.glob(str(Path(cache_root) / "test" / f"SNGS-{vid}" / "frame_*.npz")))
    rows = []
    for i, f in enumerate(files):
        if i % stride != 0:
            continue
        frame, kps = decode_npz(f)
        inl = set(ransac_inlier_keys(kps))
        rows.append({
            "video": vid,
            "idx": len(rows),
            "frame": frame,
            "image_id": f"3{vid}{frame % 1000000:06d}",
            "image_file": f"{frame % 1000000:06d}.jpg",
            "kps": kps,
            "inliers": inl,
            "outliers": set(kps) - inl,
        })
    return rows


def context(rows, idx, key, k):
    prev, nxt = [], []
    for off in range(1, k + 1):
        j = idx - off
        if j >= 0 and key in rows[j]["kps"]:
            prev.append({"offset": off, "frame": rows[j]["frame"], "point": pack(rows[j]["kps"][key])})
        j = idx + off
        if j < len(rows) and key in rows[j]["kps"]:
            nxt.append({"offset": off, "frame": rows[j]["frame"], "point": pack(rows[j]["kps"][key])})
    return prev, nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default="outputs/gsr/temporal_hrnet/round2_temporal_calib/cache_hrnet")
    ap.add_argument("--events", default="outputs/gsr/temporal_hrnet/temporal_calib_results_hub/failure_analysis/ransac_outlier_tracks_full49/outlier_events.csv")
    ap.add_argument("--out-dir", default="outputs/gsr/temporal_hrnet/temporal_calib_results_hub/failure_analysis/ransac_outlier_k5_point_cases")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--videos", type=int, default=8)
    ap.add_argument("--max-points", type=int, default=48)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_frame = defaultdict(int)
    with open(args.events, newline="") as f:
        for r in csv.DictReader(f):
            by_frame[(r["video"], int(r["idx"]))] += 1
    chosen = []
    used_videos = set()
    for (vid, idx), n in sorted(by_frame.items(), key=lambda kv: kv[1], reverse=True):
        if vid in used_videos:
            continue
        chosen.append((vid, idx, n))
        used_videos.add(vid)
        if len(chosen) >= args.videos:
            break

    cases = []
    for vid, wanted_idx, nout in chosen:
        rows = load_video(args.cache_root, vid, args.stride)
        gt = load_gt_lines_for_video(DATA_ROOT, vid)
        row = rows[wanted_idx]
        for key in sorted(row["outliers"]):
            prev, nxt = context(rows, wanted_idx, key, args.k)
            cases.append({
                "video": vid,
                "frame": row["frame"],
                "idx": wanted_idx,
                "image_id": row["image_id"],
                "image_file": row["image_file"],
                "image_path": f"datasets/SoccerNetGS/test/SNGS-{vid}/img1/{row['image_file']}",
                "frame_outlier_count": nout,
                "key": key,
                "point": pack(row["kps"][key]),
                "prev": prev,
                "next": nxt,
                "inliers": {str(k): pack(row["kps"][k]) for k in row["inliers"] if k in row["kps"]},
                "all_keypoints": {str(k): pack(v) for k, v in row["kps"].items()},
                "gt_lines": gt.get(row["image_id"], {}),
            })
            if len(cases) >= args.max_points:
                break
        if len(cases) >= args.max_points:
            break

    json.dump(cases, open(out / "point_cases_k5.json", "w"))
    with open(out / "point_cases_k5.csv", "w", newline="") as f:
        fields = ["video", "frame", "image_id", "image_file", "frame_outlier_count", "key", "n_prev", "n_next", "image_path"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in cases:
            w.writerow({k: c[k] for k in fields if k in c} | {"n_prev": len(c["prev"]), "n_next": len(c["next"])})
    print(f"wrote {len(cases)} point cases to {out}", flush=True)


if __name__ == "__main__":
    main()
