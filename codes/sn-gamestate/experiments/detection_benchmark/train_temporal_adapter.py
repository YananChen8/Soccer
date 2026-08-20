"""Train TemporalHeatmapAdapter (Plan A) on cached HRNet pred + GT heatmaps.

Data: outputs/gsr/temporal_hrnet/heatmap_cache/{split}/{video}/frame_*.npz
      (built by build_temporal_dataset.py) -> per frame: kp_hm,line_hm (frozen
      HRNet pred) + kp_gt,kp_mask,line_gt (GT, same 270x480 grid).

Windows are built WITHIN a video (no cross-video). HRNet is frozen (we train only
the adapter on cached pred heatmaps). Target = GT heatmap of the current frame.

  L = masked_MSE(H_refined, H_gt) + lam_res*||delta||^2 + lam_distill*||H_refined - H_t||^2

Run on 202:
  PY=.../wys_soccermaster/bin/python
  cd .../sn-gamestate
  PYTHONPATH=plugins/calibration:. CUDA_VISIBLE_DEVICES=1 \
    $PY experiments/detection_benchmark/train_temporal_adapter.py \
      --which kp --split valid --epochs 5 \
      --out outputs/gsr/temporal_hrnet/ckpt/kp_adapter.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from sn_gamestate.temporal_hrnet import TemporalHeatmapAdapter, pad_window

DEFAULT_CACHE = "outputs/gsr/temporal_hrnet/heatmap_cache"
WHICH = {  # which: (pred_key, gt_key, mask_key_or_None, channels)
    "kp": ("kp_hm", "kp_gt", "kp_mask", 58),
    "line": ("line_hm", "line_gt", None, 24),
}


class VideoWindowDataset(Dataset):
    """Windows within each video. Sample = (window[K,C,h,w], gt[C,h,w], mask[C])."""

    def __init__(self, split, which, window_size, cache_root, mask_mode, peak_thresh):
        self.pred_key, self.gt_key, self.mask_key, self.C = WHICH[which]
        self.k = window_size
        self.mask_mode = mask_mode
        self.peak_thresh = peak_thresh
        # ponytail: preload into RAM once (fp16). Compressed-npz decompress was
        # ~10min/epoch; in-RAM slicing makes epochs GPU-bound. Subset is small.
        self.index = []           # (vi, t)
        self.videos = []          # list per video: {pred:[N,C,h,w], gt:[N,C,h,w], mask:[N,C]}
        self.supervised_channels = []
        root = Path(cache_root) / split
        for vdir in sorted(root.iterdir()):
            if not vdir.is_dir():
                continue
            files = sorted(vdir.glob("frame_*.npz"))
            if not files:
                continue
            pred, gt, mask = [], [], []
            for f in files:
                d = np.load(f)
                pred_i = d[self.pred_key].astype(np.float16)
                gt_i = d[self.gt_key].astype(np.float16)
                base_mask = (d[self.mask_key].astype(np.float16)
                             if self.mask_key else np.ones(self.C, np.float16))
                if self.mask_mode == "present_only":
                    present = (gt_i.reshape(self.C, -1).max(axis=1) > self.peak_thresh).astype(np.float16)
                    mask_i = base_mask * present
                else:
                    mask_i = base_mask
                pred.append(pred_i)
                gt.append(gt_i)
                mask.append(mask_i)
                self.supervised_channels.append(float(mask_i.sum()))
            vi = len(self.videos)
            self.videos.append({"pred": np.stack(pred), "gt": np.stack(gt),
                                "mask": np.stack(mask)})
            for t in range(len(files)):
                self.index.append((vi, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        vi, t = self.index[i]
        v = self.videos[vi]
        lo = max(0, t - self.k + 1)
        win = torch.from_numpy(v["pred"][lo:t + 1].astype(np.float32))   # [t',C,h,w]
        win = pad_window(win.unsqueeze(0), self.k)[0]                    # [K,C,h,w]
        gt = torch.from_numpy(v["gt"][t].astype(np.float32))
        mask = torch.from_numpy(v["mask"][t].astype(np.float32))
        return win, gt, mask


def masked_mse(pred, gt, mask, fg_weight=0.0):
    # mask: [B,C] per-channel; broadcast over h,w. fg_weight upweights GT peak
    # pixels (gt>0.1) so localization error isn't drowned by background zeros.
    m = mask[:, :, None, None]
    w = m * (1.0 + fg_weight * (gt > 0.1).float())
    return ((pred - gt) ** 2 * w).sum() / (w.sum() + 1e-6)


def fg_mse(pred, gt, mask):
    """Foreground-only MSE on GT peak pixels — the metric that reflects
    localization (what the camera solve cares about). Diagnostic, not optimized."""
    fg = (gt > 0.1).float() * mask[:, :, None, None]
    return (((pred - gt) ** 2 * fg).sum() / (fg.sum() + 1e-6)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=list(WHICH), default="kp")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--eval-split", default=None, help="held-out split for generalization fg_mse")
    ap.add_argument("--cache-root", default=DEFAULT_CACHE)
    ap.add_argument("--mask-mode", choices=["dataset", "present_only"], default="dataset")
    ap.add_argument("--peak-thresh", type=float, default=0.1)
    ap.add_argument("--window-size", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam-res", type=float, default=0.01)
    ap.add_argument("--lam-distill", type=float, default=0.1)
    ap.add_argument("--fg-weight", type=float, default=50.0, help="upweight GT peak pixels")
    ap.add_argument("--residual-scale", type=float, default=1.0)
    ap.add_argument("--out", default="outputs/gsr/temporal_hrnet/ckpt/adapter.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ds = VideoWindowDataset(args.split, args.which, args.window_size,
                            args.cache_root, args.mask_mode, args.peak_thresh)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_dl = None
    if args.eval_split:
        eval_ds = VideoWindowDataset(args.eval_split, args.which, args.window_size,
                                     args.cache_root, args.mask_mode, args.peak_thresh)
        eval_dl = DataLoader(eval_ds, batch_size=args.batch_size, num_workers=0)
        print(f"held-out eval: {len(eval_ds)} windows over {len(eval_ds.videos)} videos"
              f" avg_supervised_channels={np.mean(eval_ds.supervised_channels):.2f}")
    C = WHICH[args.which][3]
    model = TemporalHeatmapAdapter(C, args.window_size,
                                   residual_scale=args.residual_scale).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"{len(ds)} windows over {len(ds.videos)} videos; which={args.which} C={C}"
          f" mask_mode={args.mask_mode} avg_supervised_channels={np.mean(ds.supervised_channels):.2f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_heldout = float("inf")

    def save(path):
        torch.save({"state_dict": model.state_dict(), "which": args.which,
                    "channels": C, "window_size": args.window_size,
                    "residual_scale": args.residual_scale,
                    "mask_mode": args.mask_mode,
                    "peak_thresh": args.peak_thresh,
                    "cache_root": args.cache_root}, path)

    for ep in range(args.epochs):
        tot, n, dmax = 0.0, 0, 0.0
        fg_base, fg_ref = 0.0, 0.0   # foreground MSE: H_t vs GT, refined vs GT
        for win, gt, mask in dl:
            win, gt, mask = win.to(args.device), gt.to(args.device), mask.to(args.device)
            refined, delta = model(win)
            l_hm = masked_mse(refined, gt, mask, args.fg_weight)
            l_res = delta.pow(2).mean()
            l_distill = (refined - win[:, -1]).pow(2).mean()
            loss = l_hm + args.lam_res * l_res + args.lam_distill * l_distill
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
            dmax = max(dmax, delta.abs().max().item())
            fg_base += fg_mse(win[:, -1], gt, mask)
            fg_ref += fg_mse(refined, gt, mask)
        msg = (f"ep{ep}: loss={tot/n:.5f} delta_max={dmax:.4f} "
               f"train_fg[H_t={fg_base/n:.5f}->ref={fg_ref/n:.5f}]")
        if eval_dl is not None:
            model.eval()
            eb, er, en = 0.0, 0.0, 0
            with torch.no_grad():
                for win, gt, mask in eval_dl:
                    win, gt, mask = win.to(args.device), gt.to(args.device), mask.to(args.device)
                    refined, _ = model(win)
                    eb += fg_mse(win[:, -1], gt, mask); er += fg_mse(refined, gt, mask); en += 1
            msg += f"  HELDOUT_fg[H_t={eb/en:.5f}->ref={er/en:.5f}]"
            model.train()
            if er / en < best_heldout:        # early-stop: keep best generalization
                best_heldout = er / en
                save(args.out.replace(".pt", "_best.pt"))
                msg += " *best"
        print(msg)

    save(args.out)
    print(f"saved final -> {args.out}; best held-out -> {args.out.replace('.pt','_best.pt')}")


if __name__ == "__main__":
    main()
