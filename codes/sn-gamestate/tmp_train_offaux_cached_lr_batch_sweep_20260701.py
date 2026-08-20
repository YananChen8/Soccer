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


class CachedShardDataset(Dataset):
    def __init__(self, manifest_path):
        self.root = Path(manifest_path).parent
        meta = json.loads(Path(manifest_path).read_text())
        self.shards = []
        self.index = []
        for si, sh in enumerate(meta["shards"]):
            path = Path(sh["path"])
            self.shards.append(path)
            for j in range(int(sh["n"])):
                self.index.append((si, j))
        self._cache_si = None
        self._cache = None

    def __len__(self):
        return len(self.index)

    def _load(self, si):
        if self._cache_si != si:
            self._cache = torch.load(self.shards[si], map_location="cpu")
            self._cache_si = si
        return self._cache

    def __getitem__(self, idx):
        si, j = self.index[idx]
        sh = self._load(si)
        return {
            "window": sh["window_u8"][j].float().div_(255.0),
            "gt": sh["gt_u8"][j].float().div_(255.0),
            "mask": sh["mask_u8"][j].float(),
            "video": sh["video"][j],
        }


def load_hrnet(device):
    cfg = yaml.safe_load(open(CFG))
    model = get_cls_net(cfg["cfg"])
    model.load_state_dict(torch.load(CKPT, map_location=device))
    return model.to(device)


def masked_mse(pred, gt, mask):
    return ((pred[:, : gt.size(1)].float() - gt).square() * mask[:, :, None, None]).mean()


def peak_xy_from_heatmap(hm):
    flat = hm.flatten(2)
    idx = flat.argmax(dim=-1)
    h, w = hm.shape[-2:]
    x = (idx % w).float() / max(w - 1, 1)
    y = (idx // w).float() / max(h - 1, 1)
    return idx, torch.stack([x, y], dim=-1)


def local_softargmax_xy(pred, gt, mask, radius_px=5.0, beta=20.0):
    pred = pred[:, :-1].float()
    gt = gt[:, :-1]
    mask = mask[:, :-1]
    b, c, h, w = pred.shape
    gt_idx, gt_xy = peak_xy_from_heatmap(gt)
    gt_conf = gt.flatten(2).amax(dim=-1)
    yy, xx = torch.meshgrid(torch.arange(h, device=pred.device), torch.arange(w, device=pred.device), indexing="ij")
    gx = (gt_idx % w).view(b, c, 1, 1)
    gy = (gt_idx // w).view(b, c, 1, 1)
    keep = ((xx.view(1, 1, h, w) - gx).square() + (yy.view(1, 1, h, w) - gy).square()) <= radius_px ** 2
    logits = (pred * beta).flatten(2).masked_fill(~keep.flatten(2), -1e4)
    prob = torch.softmax(logits, dim=-1)
    grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, h, device=pred.device), torch.linspace(0, 1, w, device=pred.device), indexing="ij")
    xy = torch.stack([(prob * grid_x.flatten()).sum(-1), (prob * grid_y.flatten()).sum(-1)], dim=-1)
    present = (mask > 0.5) & (gt_conf > 0.1)
    return xy, gt_xy, present


def aux_scaled(raw, heat, ratio):
    if ratio <= 0:
        return raw.new_zeros(())
    return raw * torch.clamp(ratio * heat.detach() / (raw.detach().abs() + 1e-8), max=200000.0)


def peak_loss(pred, gt, mask):
    xy, gt_xy, present = local_softargmax_xy(pred[:, : gt.size(1)], gt, mask)
    raw = torch.nn.functional.smooth_l1_loss(xy, gt_xy, reduction="none").sum(-1)
    return (raw * present.float()).sum() / (present.float().sum() + 1e-6)


def run_one(args, batch_size, num_workers, hrnet_lr, adapter_lr):
    random.seed(20260701)
    torch.manual_seed(20260701)
    device = torch.device(args.device)
    ds = CachedShardDataset(Path(args.cache_dir) / "cache_manifest.json")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    hrnet = load_hrnet(device)
    model = TemporalHRNetFeatureFusion(hrnet, level=args.fusion_level, window_size=3, residual_scale=args.residual_scale, freeze_hrnet=False).to(device)
    opt = torch.optim.Adam([
        {"params": model.hrnet.parameters(), "lr": hrnet_lr},
        {"params": model.fusion.parameters(), "lr": adapter_lr},
    ])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    start = time.perf_counter()
    frames = 0
    last = {}
    opt.zero_grad(set_to_none=True)
    for step, batch in enumerate(dl, 1):
        win = batch["window"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(win)
            heat = masked_mse(pred, gt, mask)
            pk = peak_loss(pred, gt, mask)
            pk_term = aux_scaled(pk, heat, 0.2)
            loss = heat + pk_term
        scaler.scale(loss).backward()
        if step % args.grad_accum_steps == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        frames += win.size(0)
        last = {"loss": float(loss.item()), "heat": float(heat.item()), "peak": float(pk.item()), "peak_term": float(pk_term.item())}
        if step >= args.max_steps:
            break
    sec = time.perf_counter() - start
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "hrnet_lr": hrnet_lr,
        "adapter_lr": adapter_lr,
        "steps": step,
        "frames": frames,
        "seconds": sec,
        "frames_per_sec": frames / max(sec, 1e-6),
        **last,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fusion-level", choices=["last", "stage1"], default="last")
    ap.add_argument("--residual-scale", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--batches", default="4,8,12")
    ap.add_argument("--workers", default="0,4,8")
    ap.add_argument("--hrnet-lrs", default="3e-6,1e-5")
    ap.add_argument("--adapter-lrs", default="3e-5,1e-4")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for bs in [int(x) for x in args.batches.split(",") if x]:
        for nw in [int(x) for x in args.workers.split(",") if x]:
            for hlr in [float(x) for x in args.hrnet_lrs.split(",") if x]:
                for alr in [float(x) for x in args.adapter_lrs.split(",") if x]:
                    try:
                        item = run_one(args, bs, nw, hlr, alr)
                        item["ok"] = True
                    except RuntimeError as exc:
                        item = {
                            "batch_size": bs,
                            "num_workers": nw,
                            "hrnet_lr": hlr,
                            "adapter_lr": alr,
                            "ok": False,
                            "error": str(exc)[:500],
                        }
                        torch.cuda.empty_cache()
                    rows.append(item)
                    (out / "speed_sweep_partial.json").write_text(json.dumps(rows, indent=2))
                    print(json.dumps(item), flush=True)
    ok = [r for r in rows if r.get("ok")]
    best = max(ok, key=lambda r: r["frames_per_sec"]) if ok else None
    result = {"rows": rows, "best": best}
    (out / "speed_sweep_results.json").write_text(json.dumps(result, indent=2))
    if rows:
        with (out / "speed_sweep_results.csv").open("w", newline="") as f:
            fields = sorted({k for r in rows for k in r})
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print("BEST", json.dumps(best, indent=2), flush=True)


if __name__ == "__main__":
    main()
