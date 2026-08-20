"""Train token temporal adapters from cached HRNet heatmaps.

Reads build_temporal_dataset.py npz files, extracts sparse keypoint tokens from
cached kp_hm, and trains only KeypointTokenTemporalAdapter. This avoids rerunning
HRNet during each training epoch.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sn_gamestate.temporal_hrnet import KeypointTokenTemporalAdapter, heatmaps_to_tokens


VIDEOS = [
    "SNGS-060", "SNGS-065", "SNGS-070", "SNGS-075", "SNGS-099", "SNGS-104",
    "SNGS-110", "SNGS-115", "SNGS-155", "SNGS-160", "SNGS-165", "SNGS-170",
]


def masked_mse(pred, gt, mask, fg_weight):
    m = mask[:, :, None, None]
    w = m * (1.0 + fg_weight * (gt > 0.1).float())
    return ((pred - gt) ** 2 * w).sum() / (w.sum() + 1e-6)


def fg_mse(pred, gt, mask):
    fg = (gt > 0.1).float() * mask[:, :, None, None]
    return (((pred - gt) ** 2 * fg).sum() / (fg.sum() + 1e-6)).item()


def pad_token_window(tokens, end, window):
    lo = max(0, end - window + 1)
    win = tokens[lo:end + 1]
    if win.shape[0] == window:
        return win
    pad = win[:1].expand(window - win.shape[0], -1, -1)
    return torch.cat([pad, win], dim=0)


def load_video(vdir, device, token_batch):
    files = sorted(Path(vdir).glob("frame_*.npz"))
    if not files:
        raise ValueError(f"no cache frames under {vdir}")
    pred, gt, mask = [], [], []
    for f in files:
        d = np.load(f)
        pred.append(d["kp_hm"].astype(np.float16))
        gt.append(d["kp_gt"].astype(np.float16))
        mask.append(d["kp_mask"].astype(np.float16))
    pred_np = np.stack(pred)
    tokens = []
    with torch.no_grad():
        for start in range(0, len(pred_np), token_batch):
            x = torch.from_numpy(pred_np[start:start + token_batch].astype(np.float32)).to(device)
            tokens.append(heatmaps_to_tokens(x).cpu().half())
            del x
    return pred_np, np.stack(gt), np.stack(mask), torch.cat(tokens, dim=0)


def train_one_video(model, opt, scaler, args, vdir, epoch, step):
    pred_np, gt_np, mask_np, tokens_cpu = load_video(vdir, args.device, args.token_batch)
    n = len(pred_np)
    order = np.arange(n)
    if args.shuffle_frames:
        np.random.shuffle(order)
    stats = {"loss": 0.0, "fg_base": 0.0, "fg_ref": 0.0, "batches": 0, "frames": 0}
    for start in range(0, n, args.batch_size):
        idx = order[start:start + args.batch_size]
        token_windows = torch.stack(
            [pad_token_window(tokens_cpu, int(i), args.window_size) for i in idx], dim=0
        ).to(args.device).float()
        current = torch.from_numpy(pred_np[idx].astype(np.float32)).to(args.device)
        gt = torch.from_numpy(gt_np[idx].astype(np.float32)).to(args.device)
        mask = torch.from_numpy(mask_np[idx].astype(np.float32)).to(args.device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            refined, delta = model(token_windows, current)
            hm_loss = masked_mse(refined.float(), gt, mask, args.fg_weight)
            res_loss = delta.float().square().mean()
            distill = (refined.float() - current.float()).square().mean()
            loss = hm_loss + args.lam_res * res_loss + args.lam_distill * distill
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        step += 1
        stats["loss"] += loss.item()
        stats["fg_base"] += fg_mse(current.float(), gt, mask)
        stats["fg_ref"] += fg_mse(refined.float(), gt, mask)
        stats["batches"] += 1
        stats["frames"] += len(idx)
        del token_windows, current, gt, mask, refined, delta
    return step, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--split", default="train12")
    ap.add_argument("--videos", nargs="+", default=VIDEOS)
    ap.add_argument("--architecture", choices=["stgcn", "transformer"], required=True)
    ap.add_argument("--window-size", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--token-batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lam-res", type=float, default=0.05)
    ap.add_argument("--lam-distill", type=float, default=0.5)
    ap.add_argument("--fg-weight", type=float, default=50.0)
    ap.add_argument("--residual-scale", type=float, default=1.0)
    ap.add_argument("--max-shift-px", type=float, default=5.0)
    ap.add_argument("--shuffle-frames", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = KeypointTokenTemporalAdapter(
        channels=58,
        window_size=args.window_size,
        architecture=args.architecture,
        hidden=args.hidden,
        residual_scale=args.residual_scale,
        max_shift_px=args.max_shift_px,
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    manifest = vars(args).copy()
    manifest["device"] = str(args.device)
    manifest["videos"] = args.videos
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    step = 0
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_stats = {"loss": 0.0, "fg_base": 0.0, "fg_ref": 0.0, "batches": 0, "frames": 0}
        for video in args.videos:
            vdir = Path(args.cache_root) / args.split / video
            step, stats = train_one_video(model, opt, scaler, args, vdir, epoch, step)
            for k in epoch_stats:
                epoch_stats[k] += stats[k]
            elapsed = time.perf_counter() - start_time
            print(
                f"epoch={epoch} video={video} step={step} "
                f"loss={stats['loss']/max(stats['batches'],1):.6f} "
                f"fg={stats['fg_base']/max(stats['batches'],1):.6f}->{stats['fg_ref']/max(stats['batches'],1):.6f} "
                f"frames/s={epoch_stats['frames']/max(elapsed,1):.2f} "
                f"gpu_mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB",
                flush=True,
            )
        ckpt = {
            "state_dict": model.state_dict(),
            "which": "kp_token",
            "channels": 58,
            "window_size": args.window_size,
            "architecture": args.architecture,
            "hidden": args.hidden,
            "residual_scale": args.residual_scale,
            "max_shift_px": args.max_shift_px,
            "videos": args.videos,
            "steps": step,
            "epoch": epoch,
            "cache_root": args.cache_root,
        }
        path = out_dir / f"kp_adapter_{args.architecture}_k{args.window_size}_ms{int(args.max_shift_px)}_epoch{epoch}.pt"
        torch.save(ckpt, path)
        print(
            f"epoch={epoch} frames={epoch_stats['frames']} "
            f"loss={epoch_stats['loss']/max(epoch_stats['batches'],1):.6f} "
            f"fg={epoch_stats['fg_base']/max(epoch_stats['batches'],1):.6f}->{epoch_stats['fg_ref']/max(epoch_stats['batches'],1):.6f} "
            f"saved={path}",
            flush=True,
        )
    final = out_dir / f"kp_adapter_{args.architecture}_k{args.window_size}_ms{int(args.max_shift_px)}.pt"
    torch.save(ckpt, final)
    print(f"saved final -> {final}")


if __name__ == "__main__":
    main()
