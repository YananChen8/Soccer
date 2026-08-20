import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from nbjw_calib.utils.utils_keypoints import KeypointsDB


DATASET_ROOT = "/remote-home/jiayuanrao/yishan/datasets/SoccerNetGS_2024_nbjw"


def read_rows(root, split, max_frames_per_video):
    by_video = defaultdict(list)
    with (Path(root) / f"{split}_manifest.tsv").open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            by_video[row["video"]].append(row)
    rows = []
    for video in sorted(by_video):
        items = sorted(by_video[video], key=lambda r: int(r["image_id"]))
        if max_frames_per_video:
            items = items[:max_frames_per_video]
        rows.extend(items)
    return rows


def keypoint_gt(json_path, image_tensor):
    data = json.load(open(json_path))
    if "Goal left post left" in data:
        data["Goal left post left "] = data.pop("Goal left post left")
    gt, mask = KeypointsDB(data, image_tensor).get_tensor_w_mask()
    return np.clip(gt * 255.0, 0, 255).astype(np.uint8), mask.astype(np.uint8)


def save_shard(out_dir, shard_idx, samples):
    path = out_dir / "shards" / f"frame_shard_{shard_idx:05d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_u8": torch.stack([s["image_u8"] for s in samples]),
        "gt_u8": torch.stack([s["gt_u8"] for s in samples]),
        "mask_u8": torch.stack([s["mask_u8"] for s in samples]),
        "video": [s["video"] for s in samples],
        "image_id": [s["image_id"] for s in samples],
        "frame_index": [s["frame_index"] for s in samples],
        "dst_image": [s["dst_image"] for s in samples],
        "dst_json": [s["dst_json"] for s in samples],
    }
    torch.save(payload, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DATASET_ROOT)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-size", type=int, default=32)
    ap.add_argument("--max-frames-per-video", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.dataset_root, args.split, args.max_frames_per_video)
    resize = T.Resize((540, 960))
    to_tensor = T.ToTensor()
    samples = []
    frame_manifest = []
    shard_manifest = []
    start = time.perf_counter()
    shard_idx = 0

    for idx, row in enumerate(rows, 1):
        image_path = Path(row["dst_image"])
        json_path = Path(row["dst_json"])
        image = resize(Image.open(image_path).convert("RGB"))
        x = to_tensor(image)
        image_u8 = (x * 255.0).round().to(torch.uint8)
        try:
            gt_u8, mask_u8 = keypoint_gt(json_path, x)
        except Exception as exc:
            print(f"WARN invalid gt {json_path}: {exc}", flush=True)
            gt_u8 = np.zeros((58, 270, 480), dtype=np.uint8)
            mask_u8 = np.zeros(58, dtype=np.uint8)

        frame_index = int(Path(row["src_image"]).stem)
        sample = {
            "image_u8": image_u8,
            "gt_u8": torch.from_numpy(gt_u8),
            "mask_u8": torch.from_numpy(mask_u8),
            "video": row["video"],
            "image_id": row["image_id"],
            "frame_index": frame_index,
            "dst_image": str(image_path),
            "dst_json": str(json_path),
        }
        samples.append(sample)
        frame_manifest.append({
            "global_index": len(frame_manifest),
            "video": row["video"],
            "image_id": row["image_id"],
            "frame_index": frame_index,
            "src_image": row["src_image"],
            "dst_image": str(image_path),
            "dst_json": str(json_path),
            "shard_index": shard_idx,
            "offset": len(samples) - 1,
        })

        if len(samples) >= args.shard_size:
            path = save_shard(out_dir, shard_idx, samples)
            shard_manifest.append({"shard_index": shard_idx, "path": str(path), "n": len(samples)})
            samples.clear()
            shard_idx += 1

        if idx % 100 == 0:
            elapsed = time.perf_counter() - start
            print(f"cached {args.split} {idx}/{len(rows)} frames elapsed={elapsed:.1f}s", flush=True)

    if samples:
        path = save_shard(out_dir, shard_idx, samples)
        shard_manifest.append({"shard_index": shard_idx, "path": str(path), "n": len(samples)})

    with (out_dir / "frames_manifest.csv").open("w", newline="") as f:
        fields = list(frame_manifest[0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frame_manifest)

    meta = {
        "dataset_root": args.dataset_root,
        "split": args.split,
        "format": "frame-level shards: image_u8 [N,3,540,960], gt_u8 [N,58,270,480], mask_u8 [N,58]. Dataset should assemble temporal windows dynamically from frames_manifest.csv.",
        "resize_hw": [540, 960],
        "heatmap_hw": [270, 480],
        "num_keypoint_channels": 58,
        "shard_size": args.shard_size,
        "max_frames_per_video": args.max_frames_per_video,
        "n_frames": len(rows),
        "n_shards": len(shard_manifest),
        "shards": shard_manifest,
    }
    (out_dir / "cache_manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"done": True, "split": args.split, "out_dir": str(out_dir), "n_frames": len(rows), "seconds": time.perf_counter() - start}, indent=2), flush=True)


if __name__ == "__main__":
    main()
