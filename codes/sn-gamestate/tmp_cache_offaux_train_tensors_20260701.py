import argparse
import csv
import json
import time
from collections import defaultdict, deque
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


def keypoint_gt(stem, image_tensor):
    data = json.load(open(str(stem) + ".json"))
    if "Goal left post left" in data:
        data["Goal left post left "] = data.pop("Goal left post left")
    gt, mask = KeypointsDB(data, image_tensor).get_tensor_w_mask()
    gt_u8 = np.clip(gt * 255.0, 0, 255).astype(np.uint8)
    return gt_u8, mask.astype(np.uint8)


def save_shard(out_dir, shard_idx, samples):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "window_u8": torch.stack([s["window_u8"] for s in samples]),
        "gt_u8": torch.stack([s["gt_u8"] for s in samples]),
        "mask_u8": torch.stack([s["mask_u8"] for s in samples]),
        "video": [s["video"] for s in samples],
        "image_id": [s["image_id"] for s in samples],
    }
    path = out_dir / f"shard_{shard_idx:05d}.pt"
    torch.save(payload, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DATASET_ROOT)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window-size", type=int, default=3)
    ap.add_argument("--shard-size", type=int, default=32)
    ap.add_argument("--max-frames-per-video", type=int, default=120)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.dataset_root, args.split, args.max_frames_per_video)
    resize = T.Resize((540, 960))
    to_tensor = T.ToTensor()
    histories = defaultdict(lambda: deque(maxlen=args.window_size))
    samples, manifest = [], []
    start = time.perf_counter()
    shard_idx = 0
    for idx, row in enumerate(rows, 1):
        stem = Path(args.dataset_root) / args.split / row["image_id"]
        image = resize(Image.open(str(stem) + ".jpg").convert("RGB"))
        x = to_tensor(image)
        x_u8 = (x * 255.0).round().to(torch.uint8)
        hist = histories[row["video"]]
        hist.append(x_u8)
        frames = list(hist)
        if len(frames) < args.window_size:
            frames = [frames[0]] * (args.window_size - len(frames)) + frames
        window_u8 = torch.stack(frames[-args.window_size:])
        try:
            gt_u8, mask_u8 = keypoint_gt(stem, x)
        except Exception as exc:
            print(f"WARN invalid gt {stem}: {exc}", flush=True)
            gt_u8 = np.zeros((58, 270, 480), dtype=np.uint8)
            mask_u8 = np.zeros(58, dtype=np.uint8)
        samples.append({
            "window_u8": window_u8,
            "gt_u8": torch.from_numpy(gt_u8),
            "mask_u8": torch.from_numpy(mask_u8),
            "video": row["video"],
            "image_id": row["image_id"],
        })
        if len(samples) >= args.shard_size:
            path = save_shard(out_dir, shard_idx, samples)
            manifest.append({"path": str(path), "n": len(samples)})
            samples.clear()
            shard_idx += 1
        if idx % 200 == 0:
            print(f"cached {idx}/{len(rows)} frames elapsed={time.perf_counter()-start:.1f}s", flush=True)
    if samples:
        path = save_shard(out_dir, shard_idx, samples)
        manifest.append({"path": str(path), "n": len(samples)})
    meta = {
        "dataset_root": args.dataset_root,
        "split": args.split,
        "window_size": args.window_size,
        "max_frames_per_video": args.max_frames_per_video,
        "shard_size": args.shard_size,
        "n_frames": len(rows),
        "format": "window_u8 [N,K,3,540,960], gt_u8 [N,58,270,480] scaled by /255, mask_u8 [N,58]",
        "shards": manifest,
    }
    (out_dir / "cache_manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"done": True, "out_dir": str(out_dir), "n_frames": len(rows), "seconds": time.perf_counter()-start}, indent=2), flush=True)


if __name__ == "__main__":
    main()
