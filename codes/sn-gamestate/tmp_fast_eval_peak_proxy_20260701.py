import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from experiments.detection_benchmark import eval_temporal_input_mixer_calib as base
from experiments.detection_benchmark.eval_temporal_feature_fusion_calib import load_feature_model
from nbjw_calib.utils.utils_heatmap import coords_to_dict, get_keypoints_from_heatmap_batch_maxpool
from nbjw_calib.utils.utils_keypoints import KeypointsDB


class EvalFrames(Dataset):
    def __init__(self, videos, stride, max_frames):
        self.rows = []
        for video in videos:
            files = sorted(Path(base.FRAMES, f"SNGS-{video}", "img1").glob("*.jpg"))
            if max_frames:
                files = files[:max_frames]
            for idx, path in enumerate(files):
                if idx % stride == 0:
                    self.rows.append((str(video), path))
        self.tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
        self.gt_cache = {str(v): base.load_gt_lines_for_video(base.DATA_ROOT, str(v)) for v in videos}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        video, path = self.rows[idx]
        pil = Image.open(path).convert("RGB")
        x = self.tfm(pil)
        gid = f"3{video}{path.stem}"
        lines = self.gt_cache[video].get(gid, {})
        try:
            gt, mask = KeypointsDB(lines or {}, x).get_tensor_w_mask()
            gt = torch.from_numpy(gt.astype(np.float32))
            mask = torch.from_numpy(mask.astype(np.float32))
        except Exception:
            gt = torch.zeros(58, 270, 480)
            mask = torch.zeros(58)
        return {"video": video, "frame": path.stem, "image": x, "gt": gt, "mask": mask}


def collate(batch):
    return {
        "video": [b["video"] for b in batch],
        "frame": [b["frame"] for b in batch],
        "image": torch.stack([b["image"] for b in batch]),
        "gt": torch.stack([b["gt"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
    }


def left_pad_batch(images, k):
    # Proxy mode does not need exact history for baseline; for fused model this repeats current frame.
    return images[:, None].repeat(1, k, 1, 1, 1)


def peak_xy(hm):
    flat = hm.flatten(2)
    idx = flat.argmax(dim=-1)
    h, w = hm.shape[-2:]
    x = (idx % w).float()
    y = (idx // w).float()
    return torch.stack([x, y], dim=-1), flat.amax(dim=-1)


def summarize_peak(pred, gt, mask, tol_px=(1, 2, 5)):
    pred = pred[:, :-1].float()
    gt = gt[:, :-1].float()
    mask = mask[:, :-1]
    pxy, pconf = peak_xy(pred)
    gxy, gconf = peak_xy(gt)
    present = (mask > 0.5) & (gconf > 0.1)
    if present.sum() == 0:
        return {"n": 0}
    dist = torch.linalg.norm((pxy - gxy) * 2.0, dim=-1)  # heatmap px -> image px
    vals = dist[present]
    out = {
        "n": int(present.sum().item()),
        "mean_dist": float(vals.mean().item()),
        "median_dist": float(vals.median().item()),
    }
    for t in tol_px:
        out[f"within_{t}px"] = float((vals <= t).float().mean().item())
    return out


def merge(items):
    total_n = sum(x.get("n", 0) for x in items)
    if total_n == 0:
        return {"n": 0}
    out = {"n": total_n}
    keys = [k for k in items[0] if k != "n"]
    for k in keys:
        out[k] = float(sum(x.get(k, 0.0) * x.get("n", 0) for x in items) / total_n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--videos", nargs="+", default=["116", "117", "118", "119", "120", "121", "122", "123"])
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    ds = EvalFrames(args.videos, args.stride, args.max_frames)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate, persistent_workers=args.workers > 0)
    kp, _line = base.load_hrnets(device)
    model = None
    window_size = 3
    if args.checkpoint:
        kp_for_model, _ = base.load_hrnets(device)
        model, window_size, _ck = load_feature_model(args.checkpoint, kp_for_model, device)
    rows = []
    by_video = defaultdict(lambda: {"baseline": [], "model": []})
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            base_hm = kp(x)
            base_s = summarize_peak(base_hm, gt, mask)
            if model is not None:
                win = left_pad_batch(x, window_size)
                model_hm = model(win)
                model_s = summarize_peak(model_hm, gt, mask)
            else:
                model_s = {}
            videos = batch["video"]
            # Batch-level row is enough for quick screening.
            row = {"videos": ",".join(sorted(set(videos))), **{f"base_{k}": v for k, v in base_s.items()}, **{f"model_{k}": v for k, v in model_s.items()}}
            rows.append(row)
            for v in sorted(set(videos)):
                idxs = [i for i, vv in enumerate(videos) if vv == v]
                by_video[v]["baseline"].append(summarize_peak(base_hm[idxs], gt[idxs], mask[idxs]))
                if model is not None:
                    by_video[v]["model"].append(summarize_peak(model_hm[idxs], gt[idxs], mask[idxs]))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": args.checkpoint,
        "videos": {},
        "aggregate": {
            "baseline": merge([x for v in by_video.values() for x in v["baseline"]]),
            "model": merge([x for v in by_video.values() for x in v["model"]]) if model is not None else {},
        },
    }
    for v, data in by_video.items():
        summary["videos"][v] = {
            "baseline": merge(data["baseline"]),
            "model": merge(data["model"]) if model is not None else {},
        }
    (out / "peak_proxy_results.json").write_text(json.dumps(summary, indent=2))
    if rows:
        with (out / "peak_proxy_batches.csv").open("w", newline="") as f:
            fields = sorted({k for r in rows for k in r})
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
