import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from experiments.detection_benchmark import eval_temporal_input_mixer_calib as base


DATA_ROOT = Path("/remote-home/jiayuanrao/yishan/sn-gamestate/data/SoccerNetGS")


def iter_frames(split, videos, stride, max_frames):
    root = DATA_ROOT / split
    video_dirs = [root / f"SNGS-{v.replace('SNGS-', '')}" for v in videos] if videos else sorted(root.glob("SNGS-*"))
    for video_dir in video_dirs:
        files = sorted((video_dir / "img1").glob("*.jpg"))
        count = 0
        for idx, path in enumerate(files):
            if idx % stride:
                continue
            if max_frames and count >= max_frames:
                break
            count += 1
            yield video_dir.name.replace("SNGS-", ""), int(path.stem), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--videos", nargs="*", default=[])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames-per-video", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--height", type=int, default=68)
    ap.add_argument("--width", type=int, default=120)
    ap.add_argument("--shard-size", type=int, default=256)
    args = ap.parse_args()

    out = Path(args.out_dir)
    data_dir = out / args.split / "shards"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / args.split / "manifest.csv"
    meta_path = out / args.split / "meta.json"

    device = torch.device(args.device)
    print(f"load_hrnets device={device}", flush=True)
    kp_model, line_model = base.load_hrnets(device)
    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    frames = list(iter_frames(args.split, args.videos, args.stride, args.max_frames_per_video))
    print(f"frames={len(frames)} split={args.split} stride={args.stride}", flush=True)

    rows = []
    shard = []
    shard_rows = []
    shard_id = 0
    t0 = time.perf_counter()

    def flush_shard():
        nonlocal shard, shard_rows, shard_id
        if not shard:
            return
        shard_path = data_dir / f"shard_{shard_id:05d}.npy"
        np.save(shard_path, np.stack(shard).astype(np.float16))
        rel = str(shard_path.relative_to(out))
        for offset, row in enumerate(shard_rows):
            row["path"] = rel
            row["offset"] = offset
            rows.append(row)
        print(f"wrote_shard={shard_id} frames={len(shard)} total={len(rows)}", flush=True)
        shard = []
        shard_rows = []
        shard_id += 1

    for start in range(0, len(frames), args.batch_size):
        batch = frames[start:start + args.batch_size]
        imgs = torch.stack([tfm(Image.open(p).convert("RGB")) for _, _, p in batch]).to(device)
        with torch.no_grad():
            kp = kp_model(imgs).float()
            line = line_model(imgs).float()
            hm = torch.cat([kp, line], dim=1)
            hm = F.interpolate(hm, size=(args.height, args.width), mode="bilinear", align_corners=False)
            hm = hm.cpu().numpy().astype(np.float16)
        for i, (video, frame, image_path) in enumerate(batch):
            shard.append(hm[i])
            shard_rows.append({
                "split": args.split,
                "video": video,
                "frame": frame,
                "path": "",
                "offset": -1,
                "image_path": str(image_path),
            })
            if len(shard) >= args.shard_size:
                flush_shard()
        if (start // args.batch_size) % 10 == 0:
            print(f"cached={min(start + len(batch), len(frames))}/{len(frames)}", flush=True)
    flush_shard()

    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "video", "frame", "path", "offset", "image_path"])
        w.writeheader()
        w.writerows(rows)
    meta_path.write_text(json.dumps({
        "split": args.split,
        "videos": args.videos,
        "stride": args.stride,
        "max_frames_per_video": args.max_frames_per_video,
        "height": args.height,
        "width": args.width,
        "channels": int(hm.shape[1]) if rows else None,
        "shard_size": args.shard_size,
        "frames": len(rows),
        "seconds": time.perf_counter() - t0,
    }, indent=2))
    print(f"done rows={len(rows)} seconds={time.perf_counter() - t0:.1f} out={out}", flush=True)


if __name__ == "__main__":
    main()
