import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from nbjw_calib.model.cls_hrnet import get_cls_net
from sn_gamestate.temporal_hrnet import TemporalHRNetFeatureFusion


CFG = "sn_gamestate/configs/modules/pitch/nbjw_calib.yaml"
CKPT = "/remote-home/jiayuanrao/yishan/sn-gamestate/pretrained_models/calibration/SV_kp"
H, W = 540, 960
HM_H, HM_W = 270, 480
KPTS = 58


class FrameMemmapWindowDataset(Dataset):
    def __init__(self, cache_dir, k=3, max_samples=4096):
        self.cache_dir = Path(cache_dir)
        meta = json.loads((self.cache_dir / "cache_meta.json").read_text())
        self.full_shape = tuple(meta.get("image_shape", [meta.get("n_frames"), 3, H, W]))
        self.n = int(meta.get("frames", meta.get("n_frames", self.full_shape[0])))
        if max_samples and max_samples < self.n:
            self.n = int(max_samples)
        self.k = int(k)
        self.img_path = self.cache_dir / f"images_u8_chw_{H}x{W}.dat"
        lab = np.load(self.cache_dir / "compact_labels.npz")
        self.coords = lab["coords"][: self.n]
        self.masks = lab["masks"][: self.n]
        self.images = None

    def _images(self):
        if self.images is None:
            self.images = np.memmap(self.img_path, mode="r", dtype=np.uint8, shape=self.full_shape)
        return self.images

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        imgs = self._images()
        window = []
        for off in range(self.k - 1, -1, -1):
            j = max(0, idx - off)
            arr = np.asarray(imgs[j])
            window.append(torch.from_numpy(arr.copy()).float().div_(255.0))
        return {
            "window": torch.stack(window),
            "coords": torch.from_numpy(self.coords[idx].astype(np.int64)),
            "mask": torch.from_numpy(self.masks[idx].astype(np.float32)),
        }


def load_model(device, fusion_level, residual_scale):
    cfg = yaml.safe_load(open(CFG))
    hrnet = get_cls_net(cfg["cfg"])
    hrnet.load_state_dict(torch.load(CKPT, map_location=device))
    model = TemporalHRNetFeatureFusion(
        hrnet,
        level=fusion_level,
        window_size=3,
        residual_scale=residual_scale,
        freeze_hrnet=False,
    )
    return model.to(device)


def coords_to_heatmap(coords, mask, sigma=1.5):
    b = coords.shape[0]
    yy, xx = torch.meshgrid(
        torch.arange(HM_H, device=coords.device),
        torch.arange(HM_W, device=coords.device),
        indexing="ij",
    )
    x = coords[:, :, 0].view(b, KPTS, 1, 1).float()
    y = coords[:, :, 1].view(b, KPTS, 1, 1).float()
    valid = (mask > 0.5) & (coords[:, :, 0] >= 0) & (coords[:, :, 1] >= 0)
    dist2 = (xx.view(1, 1, HM_H, HM_W) - x).square() + (yy.view(1, 1, HM_H, HM_W) - y).square()
    hm = torch.exp(-dist2 / (2.0 * sigma * sigma))
    return hm * valid[:, :, None, None].float(), valid.float()


def masked_mse(pred, gt, mask):
    pred = pred[:, : gt.shape[1]].float()
    return ((pred - gt).square() * mask[:, :, None, None]).sum() / (
        mask.sum() * gt.shape[-1] * gt.shape[-2] + 1e-6
    )


def run_one(args, batch_size, workers):
    random.seed(20260701)
    np.random.seed(20260701)
    torch.manual_seed(20260701)
    device = torch.device(args.device)
    ds = FrameMemmapWindowDataset(args.cache_dir, k=3, max_samples=args.max_samples)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    model = load_model(device, args.fusion_level, args.residual_scale)
    opt = torch.optim.Adam(
        [
            {"params": model.hrnet.parameters(), "lr": args.hrnet_lr},
            {"params": model.fusion.parameters(), "lr": args.adapter_lr},
        ]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    last = {}
    frames = 0
    opt.zero_grad(set_to_none=True)
    start = time.perf_counter()
    for step, batch in enumerate(dl, 1):
        win = batch["window"].to(device, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        mask0 = batch["mask"].to(device, non_blocking=True)
        with torch.no_grad():
            gt, mask = coords_to_heatmap(coords, mask0, sigma=args.sigma)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(win)
            loss = masked_mse(pred, gt, mask)
        scaler.scale(loss).backward()
        if step % args.grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        frames += int(win.shape[0])
        last = {"loss": float(loss.detach().cpu().item())}
        if device.type == "cuda":
            torch.cuda.synchronize()
        if step >= args.steps:
            break
    sec = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    del model, opt, dl, ds
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "batch_size": batch_size,
        "num_workers": workers,
        "steps": step,
        "frames": frames,
        "seconds": sec,
        "frames_per_sec": frames / max(sec, 1e-6),
        "peak_cuda_gb": peak_mem,
        "hrnet_lr": args.hrnet_lr,
        "adapter_lr": args.adapter_lr,
        **last,
        "ok": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fusion-level", default="last", choices=["last", "stage1"])
    ap.add_argument("--residual-scale", type=float, default=0.05)
    ap.add_argument("--hrnet-lr", type=float, default=3e-6)
    ap.add_argument("--adapter-lr", type=float, default=3e-5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max-samples", type=int, default=4096)
    ap.add_argument("--batches", default="4,8,12,16")
    ap.add_argument("--workers", default="0,4,8")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=1.5)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for bs in [int(x) for x in args.batches.split(",") if x]:
        for nw in [int(x) for x in args.workers.split(",") if x]:
            try:
                item = run_one(args, bs, nw)
            except RuntimeError as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                item = {"batch_size": bs, "num_workers": nw, "ok": False, "error": str(exc)[:800]}
            rows.append(item)
            (out / "speed_sweep_partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(json.dumps(item), flush=True)

    ok = [r for r in rows if r.get("ok")]
    best = max(ok, key=lambda r: r["frames_per_sec"]) if ok else None
    result = {"best": best, "rows": rows}
    (out / "speed_sweep_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out / "speed_sweep_results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("BEST", json.dumps(best, indent=2), flush=True)


if __name__ == "__main__":
    main()
