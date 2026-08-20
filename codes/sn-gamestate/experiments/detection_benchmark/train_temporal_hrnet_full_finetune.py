"""Full-finetune NBJW keypoint HRNet with minimal temporal feature fusion."""
import argparse
import csv
import json
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image

from nbjw_calib.model.cls_hrnet import get_cls_net
from nbjw_calib.utils.utils_keypoints import KeypointsDB
from sn_gamestate.temporal_hrnet import TemporalHRNetFeatureFusion


DATASET_ROOT = "/remote-home/jiayuanrao/yishan/datasets/SoccerNetGS_2024_nbjw"
CFG = "sn_gamestate/configs/modules/pitch/nbjw_calib.yaml"
CKPT = "/remote-home/jiayuanrao/yishan/sn-gamestate/pretrained_models/calibration/SV_kp"


def read_videos(dataset_root, split):
    videos = defaultdict(list)
    with (Path(dataset_root) / f"{split}_manifest.tsv").open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            stem = Path(dataset_root) / split / row["image_id"]
            videos[row["video"]].append((int(row["image_id"]), stem))
    for frames in videos.values():
        frames.sort()
    return dict(videos)


def load_hrnet(device):
    cfg = yaml.safe_load(open(CFG))
    model = get_cls_net(cfg["cfg"])
    model.load_state_dict(torch.load(CKPT, map_location=device))
    return model.to(device)


def load_frame(stem, resize, to_tensor, peak_thresh):
    image = resize(Image.open(str(stem) + ".jpg").convert("RGB"))
    x = to_tensor(image)
    data = json.load(open(str(stem) + ".json"))
    if "Goal left post left" in data:
        data["Goal left post left "] = data.pop("Goal left post left")
    try:
        gt, mask = KeypointsDB(data, x).get_tensor_w_mask()
        gt = gt.astype(np.float32)
        mask = mask.astype(np.float32)
        present = (gt.reshape(58, -1).max(axis=1) > peak_thresh).astype(np.float32)
        mask *= present
    except Exception:
        print(f"WARNING: invalid keypoint GT for {stem}; masking this frame", flush=True)
        traceback.print_exc()
        gt = np.zeros((58, 270, 480), dtype=np.float32)
        mask = np.zeros(58, dtype=np.float32)
    return x, torch.from_numpy(gt), torch.from_numpy(mask)


def left_pad_window(frames, window_size):
    if len(frames) >= window_size:
        return torch.stack(frames[-window_size:])
    return torch.stack([frames[0]] * (window_size - len(frames)) + list(frames))


def masked_mse(pred, gt, mask, fg_weight):
    """Official-style masked L2 over HRNet softmax heatmaps.

    NBJW keypoint targets include a final background channel. Do not apply
    foreground boosting to that channel: it is near one over most pixels and
    would dominate the gradient while being excluded by keypoint decoding.
    """
    pred = pred[:, : gt.size(1)].float()
    channel_mask = mask[:, :, None, None]
    weight = channel_mask
    if fg_weight > 0:
        fg = torch.zeros_like(gt)
        fg[:, :-1] = (gt[:, :-1] > 0.1).float()
        weight = channel_mask * (1.0 + fg_weight * fg)
    return ((pred - gt).square() * weight).sum() / (weight.sum() + 1e-6)


def peak_xy_from_heatmap(hm):
    flat = hm.flatten(2)
    idx = flat.argmax(dim=-1)
    h, w = hm.shape[-2:]
    x = (idx % w).float() / max(w - 1, 1)
    y = (idx // w).float() / max(h - 1, 1)
    return idx, torch.stack([x, y], dim=-1)


def local_softargmax_xy(pred, gt, mask, radius_px=5.0, beta=20.0, gt_conf_th=0.1):
    pred = pred[:, : gt.size(1)].float()
    pred = pred[:, :-1]
    gt = gt[:, :-1]
    mask = mask[:, :-1]
    b, c, h, w = pred.shape
    gt_idx, gt_xy = peak_xy_from_heatmap(gt)
    gt_conf = gt.flatten(2).amax(dim=-1)
    yy, xx = torch.meshgrid(
        torch.arange(h, device=pred.device),
        torch.arange(w, device=pred.device),
        indexing="ij",
    )
    gx = (gt_idx % w).view(b, c, 1, 1)
    gy = (gt_idx // w).view(b, c, 1, 1)
    keep = (xx.view(1, 1, h, w) - gx).square() + (yy.view(1, 1, h, w) - gy).square()
    keep = keep <= float(radius_px) ** 2
    logits = pred.flatten(2) * beta
    logits = logits.masked_fill(~keep.flatten(2), -1e4)
    prob = torch.softmax(logits, dim=-1)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, h, device=pred.device, dtype=pred.dtype),
        torch.linspace(0, 1, w, device=pred.device, dtype=pred.dtype),
        indexing="ij",
    )
    xy = torch.stack(
        [(prob * grid_x.flatten()).sum(-1), (prob * grid_y.flatten()).sum(-1)],
        dim=-1,
    )
    present = (mask > 0.5) & (gt_conf > gt_conf_th)
    return xy, gt_xy, present


def peak_location_loss(pred, gt, mask, radius_px, beta, gt_conf_th):
    xy, gt_xy, present = local_softargmax_xy(pred, gt, mask, radius_px, beta, gt_conf_th)
    raw = torch.nn.functional.smooth_l1_loss(xy, gt_xy, reduction="none").sum(-1)
    return (raw * present.float()).sum() / (present.float().sum() + 1e-6)


def temporal_motion_loss(pred, gt, mask, video_names, radius_px, beta, gt_conf_th):
    xy, gt_xy, present = local_softargmax_xy(pred, gt, mask, radius_px, beta, gt_conf_th)
    terms = []
    for i in range(1, pred.size(0)):
        if video_names[i] != video_names[i - 1]:
            continue
        keep = present[i] & present[i - 1]
        if keep.any():
            pred_delta = xy[i] - xy[i - 1]
            gt_delta = gt_xy[i] - gt_xy[i - 1]
            terms.append(torch.nn.functional.smooth_l1_loss(pred_delta[keep], gt_delta[keep]))
    return torch.stack(terms).mean() if terms else pred.new_zeros(())


def fg_mse(pred, gt, mask):
    pred = pred[:, : gt.size(1)].float()
    fg = torch.zeros_like(gt)
    fg[:, :-1] = (gt[:, :-1] > 0.1).float()
    fg = fg * mask[:, :, None, None]
    return (((pred - gt).square() * fg).sum() / (fg.sum() + 1e-6)).item()


def train_batch(pending, model, optimizer, scaler, device, use_amp, args, step, totals):
    windows = torch.stack([item[0] for item in pending]).to(device)
    gt = torch.stack([item[1] for item in pending]).to(device)
    mask = torch.stack([item[2] for item in pending]).to(device)
    video_names = [item[3] for item in pending]

    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        pred = model(windows)
        heat = masked_mse(pred, gt, mask, args.fg_weight)
        peak = peak_location_loss(pred, gt, mask, args.peak_radius_px, args.peak_beta, args.gt_conf_th)
        motion = temporal_motion_loss(pred, gt, mask, video_names, args.peak_radius_px, args.peak_beta, args.gt_conf_th)
        loss = heat + args.peak_weight * peak + args.motion_weight * motion

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    step += 1
    totals["loss"] += loss.item()
    totals["heat"] += heat.item()
    totals["peak"] += peak.item()
    totals["motion"] += motion.item()
    totals["fg"] += fg_mse(pred, gt, mask)
    totals["batches"] += 1
    totals["frames"] += len(pending)
    pending.clear()
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DATASET_ROOT)
    ap.add_argument("--split", default="train")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fusion-level", choices=["stage1", "last"], required=True)
    ap.add_argument("--window-size", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--hrnet-lr", type=float, default=1e-5)
    ap.add_argument("--adapter-lr", type=float, default=3e-5)
    ap.add_argument("--fg-weight", type=float, default=0.0)
    ap.add_argument("--peak-weight", type=float, default=0.03)
    ap.add_argument("--motion-weight", type=float, default=0.0)
    ap.add_argument("--peak-radius-px", type=float, default=5.0)
    ap.add_argument("--peak-beta", type=float, default=20.0)
    ap.add_argument("--gt-conf-th", type=float, default=0.1)
    ap.add_argument("--residual-scale", type=float, default=0.05)
    ap.add_argument("--peak-thresh", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    videos = read_videos(args.dataset_root, args.split)
    video_names = list(videos)
    if args.videos:
        missing = [video for video in args.videos if video not in videos]
        if missing:
            raise ValueError(f"videos not found in {args.split}: {missing}")
        video_names = list(args.videos)
    if args.max_videos:
        video_names = video_names[: args.max_videos]

    hrnet = load_hrnet(device)
    model = TemporalHRNetFeatureFusion(
        hrnet,
        level=args.fusion_level,
        window_size=args.window_size,
        residual_scale=args.residual_scale,
        freeze_hrnet=False,
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.hrnet.parameters(), "lr": args.hrnet_lr},
            {"params": model.fusion.parameters(), "lr": args.adapter_lr},
        ]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    resize = T.Resize((540, 960))
    to_tensor = T.ToTensor()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    def checkpoint_payload(epoch, steps):
        return {
            "state_dict": model.state_dict(),
            "model": "TemporalHRNetFullFineTune",
            "fusion_level": args.fusion_level,
            "window_size": args.window_size,
            "residual_scale": args.residual_scale,
            "full_finetune": True,
            "loss_weights": {"heat": 1.0, "peak": args.peak_weight, "motion": args.motion_weight},
            "hrnet_lr": args.hrnet_lr,
            "adapter_lr": args.adapter_lr,
            "split": args.split,
            "videos": video_names,
            "epoch": epoch,
            "steps": steps,
        }

    print(
        f"full-finetune split={args.split} level={args.fusion_level} videos={len(video_names)} "
        f"window={args.window_size} batch={args.batch_size} peak={args.peak_weight} motion={args.motion_weight}",
        flush=True,
    )
    global_step = 0
    run_start = time.perf_counter()
    use_amp = device.type == "cuda"
    for epoch in range(args.epochs):
        random.shuffle(video_names)
        pending = []
        totals = {"loss": 0.0, "heat": 0.0, "peak": 0.0, "motion": 0.0, "fg": 0.0, "batches": 0, "frames": 0}
        for video in video_names:
            history = []
            frames = videos[video][: args.max_frames or None]
            for _image_id, stem in frames:
                image, gt, mask = load_frame(stem, resize, to_tensor, args.peak_thresh)
                history.append(image)
                if len(history) > args.window_size:
                    history.pop(0)
                pending.append((left_pad_window(history, args.window_size), gt, mask, video))
                if len(pending) >= args.batch_size:
                    before = global_step
                    global_step = train_batch(pending, model, optimizer, scaler, device, use_amp, args, global_step, totals)
                    if args.log_every and global_step != before and global_step % args.log_every == 0:
                        elapsed = time.perf_counter() - run_start
                        denom = max(totals["batches"], 1)
                        print(
                            f"epoch={epoch} step={global_step} "
                            f"loss={totals['loss']/denom:.6f} heat={totals['heat']/denom:.6f} "
                            f"peak={totals['peak']/denom:.6f} motion={totals['motion']/denom:.6f} "
                            f"fg={totals['fg']/denom:.6f} frames/s={totals['frames']/max(elapsed,1):.2f}",
                            flush=True,
                        )
                if args.max_steps and global_step >= args.max_steps:
                    break
            if pending:
                global_step = train_batch(pending, model, optimizer, scaler, device, use_amp, args, global_step, totals)
            if args.max_steps and global_step >= args.max_steps:
                break
        epoch_path = args.out.replace(".pt", f"_epoch{epoch + 1}.pt")
        torch.save(checkpoint_payload(epoch, global_step), epoch_path)
        torch.save(checkpoint_payload(epoch, global_step), args.out)
        denom = max(totals["batches"], 1)
        print(
            f"epoch={epoch} frames={totals['frames']} loss={totals['loss']/denom:.6f} "
            f"heat={totals['heat']/denom:.6f} peak={totals['peak']/denom:.6f} "
            f"motion={totals['motion']/denom:.6f} fg={totals['fg']/denom:.6f} saved={epoch_path}",
            flush=True,
        )
        if args.max_steps and global_step >= args.max_steps:
            break
    torch.save(checkpoint_payload(epoch, global_step), args.out)
    print(f"saved final -> {args.out}; steps={global_step}", flush=True)


if __name__ == "__main__":
    main()
