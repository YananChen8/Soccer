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
from torch.utils.data import DataLoader, Dataset

from nbjw_calib.utils.utils_keypoints import KeypointsDB


DATASET_ROOT = Path("/remote-home/jiayuanrao/yishan/datasets/SoccerNetGS_2024_nbjw")
OUT = Path(
    "outputs/gsr/temporal_hrnet/temporal_calib_results_hub/"
    "full_finetune_temporal_nbjw_k3_official_aux_20260701/data_format_benchmark_20260701"
)
H, W = 540, 960
HM_H, HM_W = 270, 480
KPTS = 58


def read_rows(split, max_frames, max_videos):
    by_video = defaultdict(list)
    with (DATASET_ROOT / f"{split}_manifest.tsv").open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            by_video[row["video"]].append(row)
    rows = []
    for video in sorted(by_video)[:max_videos]:
        items = sorted(by_video[video], key=lambda r: int(r["image_id"]))
        rows.extend(items)
        if len(rows) >= max_frames:
            break
    return rows[:max_frames]


def make_gt(json_path, image_tensor):
    data = json.load(open(json_path))
    if "Goal left post left" in data:
        data["Goal left post left "] = data.pop("Goal left post left")
    gt, mask = KeypointsDB(data, image_tensor).get_tensor_w_mask()
    coords = np.full((KPTS, 2), -1, dtype=np.int16)
    gt = np.asarray(gt)
    mask = np.asarray(mask).astype(np.uint8)
    for c in range(min(KPTS, gt.shape[0])):
        if mask[c] == 0 or gt[c].max() <= 0:
            continue
        y, x = np.unravel_index(int(gt[c].argmax()), gt[c].shape)
        coords[c] = (x, y)
    return coords, mask


def build_sample_cache(rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"images_u8_n{len(rows)}_chw_{H}x{W}.dat"
    coords_path = out_dir / "compact_labels.npz"
    manifest_path = out_dir / "manifest.csv"
    images = np.memmap(img_path, mode="w+", dtype=np.uint8, shape=(len(rows), 3, H, W))
    coords = np.full((len(rows), KPTS, 2), -1, dtype=np.int16)
    masks = np.zeros((len(rows), KPTS), dtype=np.uint8)
    resize = T.Resize((H, W))
    to_tensor = T.ToTensor()
    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        image = resize(Image.open(row["dst_image"]).convert("RGB"))
        x = to_tensor(image)
        images[i] = (x.numpy() * 255.0).round().astype(np.uint8)
        try:
            coords[i], masks[i] = make_gt(row["dst_json"], x)
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"cache {i+1}/{len(rows)} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    images.flush()
    np.savez_compressed(coords_path, coords=coords, masks=masks)
    with manifest_path.open("w", newline="") as f:
        fields = ["idx", "video", "image_id", "dst_image", "dst_json"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(rows):
            w.writerow({k: row.get(k, "") for k in fields if k != "idx"} | {"idx": i})
    return img_path, coords_path, manifest_path, time.perf_counter() - t0


class JpgWindowDataset(Dataset):
    def __init__(self, rows, coords, masks, k=3):
        self.rows, self.coords, self.masks, self.k = rows, coords, masks, k
        self.resize = T.Resize((H, W))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        imgs = []
        for off in range(self.k - 1, -1, -1):
            j = max(0, idx - off)
            image = self.resize(Image.open(self.rows[j]["dst_image"]).convert("RGB"))
            imgs.append(self.to_tensor(image))
        return torch.stack(imgs), torch.from_numpy(self.coords[idx]), torch.from_numpy(self.masks[idx])


class MemmapWindowDataset(Dataset):
    def __init__(self, img_path, n, coords, masks, k=3):
        self.img_path, self.n, self.coords, self.masks, self.k = str(img_path), n, coords, masks, k
        self.images = None

    def _images(self):
        if self.images is None:
            self.images = np.memmap(self.img_path, mode="r", dtype=np.uint8, shape=(self.n, 3, H, W))
        return self.images

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        arr = self._images()
        imgs = []
        for off in range(self.k - 1, -1, -1):
            j = max(0, idx - off)
            imgs.append(torch.from_numpy(np.asarray(arr[j])).float().div_(255.0))
        return torch.stack(imgs), torch.from_numpy(self.coords[idx]), torch.from_numpy(self.masks[idx])


def bench_loader(name, ds, workers, batch_size, steps, device):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    t0 = time.perf_counter()
    n = 0
    for step, batch in enumerate(loader, 1):
        x = batch[0]
        if device != "cpu":
            x = x.to(device, non_blocking=True)
            torch.cuda.synchronize()
        n += x.shape[0]
        if step >= steps:
            break
    sec = time.perf_counter() - t0
    return {"name": name, "workers": workers, "batch_size": batch_size, "steps": step, "samples": n, "seconds": sec, "samples_per_s": n / sec}


def bench_serial_current(rows, n):
    resize = T.Resize((H, W))
    to_tensor = T.ToTensor()
    t0 = time.perf_counter()
    ok = 0
    for row in rows[:n]:
        image = resize(Image.open(row["dst_image"]).convert("RGB"))
        x = to_tensor(image)
        try:
            make_gt(row["dst_json"], x)
        except Exception:
            pass
        ok += 1
    sec = time.perf_counter() - t0
    return {"name": "serial_jpg_keypointsdb", "workers": 0, "batch_size": 1, "steps": ok, "samples": ok, "seconds": sec, "samples_per_s": ok / sec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=1024)
    ap.add_argument("--max-videos", type=int, default=4)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_rows("train", args.max_frames, args.max_videos)
    img_path, coords_path, manifest_path, build_seconds = build_sample_cache(rows, OUT / "sample_memmap_compact")
    lab = np.load(coords_path)
    coords, masks = lab["coords"], lab["masks"]
    results = [bench_serial_current(rows, min(256, len(rows)))]
    for workers in [0, 4, 8, 16]:
        results.append(bench_loader("jpg_compact_labels", JpgWindowDataset(rows, coords, masks), workers, args.batch_size, args.steps, args.device))
        results.append(bench_loader("memmap_u8_compact_labels", MemmapWindowDataset(img_path, len(rows), coords, masks), workers, args.batch_size, args.steps, args.device))
    total_frames = 79500
    bytes_per_frame = 3 * H * W
    estimate = {
        "memmap_image_u8_full_dataset_gb": total_frames * bytes_per_frame / 1024**3,
        "compact_labels_full_dataset_gb": total_frames * (KPTS * 2 * 2 + KPTS) / 1024**3,
        "target_under_400gb": True,
        "sample_frames": len(rows),
        "sample_memmap_bytes": os.path.getsize(img_path),
        "sample_build_seconds": build_seconds,
    }
    with (OUT / "data_format_benchmark_results.json").open("w") as f:
        json.dump({"results": results, "estimate": estimate}, f, indent=2)
    with (OUT / "data_format_benchmark_results.csv").open("w", newline="") as f:
        fields = list(results[0])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    best = max(results, key=lambda r: r["samples_per_s"])
    (OUT / "RECOMMENDED_DATA_FORMAT.md").write_text(
        "# Recommended Data Format\n\n"
        f"Best sampled loader: `{best['name']}` workers={best['workers']} batch={best['batch_size']} "
        f"throughput={best['samples_per_s']:.2f} samples/s.\n\n"
        "Recommended full format: contiguous uint8 image memmap `[N,3,540,960]` plus compact keypoint labels "
        "`coords int16 [N,58,2]` and `mask uint8 [N,58]`. It avoids JPEG decode/resize and avoids dense heatmap storage.\n\n"
        f"Estimated full image cache: {estimate['memmap_image_u8_full_dataset_gb']:.1f} GB; "
        f"compact labels: {estimate['compact_labels_full_dataset_gb']:.3f} GB; total below 400 GB.\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(OUT), "best": best, "estimate": estimate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
