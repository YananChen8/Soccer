import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from nbjw_calib.utils.utils_keypoints import KeypointsDB


DATASET_ROOT = Path("/remote-home/jiayuanrao/yishan/datasets/SoccerNetGS_2024_nbjw")
H, W = 540, 960
KPTS = 58


def read_rows(root, split, max_frames=0):
    by_video = defaultdict(list)
    with (Path(root) / f"{split}_manifest.tsv").open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            by_video[row["video"]].append(row)
    rows = []
    for video in sorted(by_video):
        rows.extend(sorted(by_video[video], key=lambda r: int(r["image_id"])))
    return rows[:max_frames] if max_frames else rows


def make_gt(json_path, image_tensor):
    data = json.load(open(json_path))
    if "Goal left post left" in data:
        data["Goal left post left "] = data.pop("Goal left post left")
    gt, mask = KeypointsDB(data, image_tensor).get_tensor_w_mask()
    gt = np.asarray(gt)
    mask = np.asarray(mask).astype(np.uint8)
    coords = np.full((KPTS, 2), -1, dtype=np.int16)
    for c in range(min(KPTS, gt.shape[0])):
        if c >= len(mask) or mask[c] == 0 or gt[c].max() <= 0:
            continue
        y, x = np.unravel_index(int(gt[c].argmax()), gt[c].shape)
        coords[c] = (x, y)
    return coords, mask[:KPTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=str(DATASET_ROOT))
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--flush-every", type=int, default=500)
    args = ap.parse_args()

    rows = read_rows(args.dataset_root, args.split, args.max_frames)
    out_dir = Path(args.out_root) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"images_u8_chw_{H}x{W}.dat"
    labels_path = out_dir / "compact_labels.npz"
    manifest_path = out_dir / "manifest.csv"
    meta_path = out_dir / "cache_meta.json"
    progress_path = out_dir / "progress.json"

    n = len(rows)
    images = np.memmap(img_path, mode="w+", dtype=np.uint8, shape=(n, 3, H, W))
    coords = np.full((n, KPTS, 2), -1, dtype=np.int16)
    masks = np.zeros((n, KPTS), dtype=np.uint8)

    resize = T.Resize((H, W))
    to_tensor = T.ToTensor()
    start = time.perf_counter()
    errors = []

    with manifest_path.open("w", newline="") as mf:
        fields = ["idx", "video", "image_id", "frame_index", "src_image", "dst_image", "dst_json"]
        writer = csv.DictWriter(mf, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            image_path = Path(row["dst_image"])
            json_path = Path(row["dst_json"])
            try:
                image = resize(Image.open(image_path).convert("RGB"))
                x = to_tensor(image)
                images[i] = (x.numpy() * 255.0).round().astype(np.uint8)
                coords[i], masks[i] = make_gt(json_path, x)
            except Exception as exc:
                errors.append({"idx": i, "dst_image": str(image_path), "dst_json": str(json_path), "error": str(exc)})
                if len(errors) <= 20:
                    print(f"WARN {args.split} idx={i} image={image_path} error={exc}", flush=True)
            frame_index = Path(row.get("src_image", "")).stem
            writer.writerow({
                "idx": i,
                "video": row.get("video", ""),
                "image_id": row.get("image_id", ""),
                "frame_index": frame_index,
                "src_image": row.get("src_image", ""),
                "dst_image": str(image_path),
                "dst_json": str(json_path),
            })
            if (i + 1) % args.flush_every == 0:
                images.flush()
                elapsed = time.perf_counter() - start
                progress = {
                    "split": args.split,
                    "done": i + 1,
                    "total": n,
                    "elapsed_seconds": elapsed,
                    "frames_per_second": (i + 1) / elapsed,
                    "estimated_remaining_seconds": (n - i - 1) / max((i + 1) / elapsed, 1e-9),
                    "image_bytes": os.path.getsize(img_path),
                    "error_count": len(errors),
                }
                progress_path.write_text(json.dumps(progress, indent=2))
                print(json.dumps(progress), flush=True)

    images.flush()
    np.savez_compressed(labels_path, coords=coords, masks=masks)
    if errors:
        (out_dir / "errors.json").write_text(json.dumps(errors, indent=2))
    meta = {
        "format": "uint8 image memmap [N,3,540,960] + compact labels coords int16 [N,58,2], masks uint8 [N,58]",
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "n_frames": n,
        "image_shape": [n, 3, H, W],
        "image_dtype": "uint8",
        "coords_dtype": "int16",
        "mask_dtype": "uint8",
        "image_path": str(img_path),
        "labels_path": str(labels_path),
        "manifest_path": str(manifest_path),
        "seconds": time.perf_counter() - start,
        "error_count": len(errors),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    progress_path.write_text(json.dumps({**meta, "done": n, "total": n}, indent=2))
    print(json.dumps({"done": True, **meta}, indent=2), flush=True)


if __name__ == "__main__":
    main()
