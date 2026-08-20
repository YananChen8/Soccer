"""Smoke-train feature-level temporal fusion inside frozen NBJW HRNet."""
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
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


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
    channel_mask = mask[:, :, None, None]
    weight = channel_mask * (1.0 + fg_weight * (gt > 0.1).float())
    return ((pred - gt).square() * weight).sum() / (weight.sum() + 1e-6)


def fg_mse(pred, gt, mask):
    fg = (gt > 0.1).float() * mask[:, :, None, None]
    return (((pred - gt).square() * fg).sum() / (fg.sum() + 1e-6)).item()


def softargmax_xy(pred, beta=1000.0):
    hm = pred[:, :-1].flatten(2)
    prob = torch.softmax(hm * beta, dim=-1)
    h, w = pred.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h, device=pred.device, dtype=pred.dtype),
        torch.linspace(0, 1, w, device=pred.device, dtype=pred.dtype),
        indexing="ij",
    )
    x = (prob * xx.flatten()).sum(-1)
    y = (prob * yy.flatten()).sum(-1)
    return torch.stack([x, y], dim=-1)


def continuity_loss(pred, video_names):
    # ponytail: proxy for camera continuity; NBJW decode/RANSAC is not differentiable.
    coords = softargmax_xy(pred)
    terms = [
        (coords[i] - coords[i - 1]).square().mean()
        for i in range(1, pred.size(0))
        if video_names[i] == video_names[i - 1]
    ]
    if not terms:
        return pred.new_zeros(())
    return torch.stack(terms).mean()


def peak_xy_from_heatmap(hm):
    flat = hm.flatten(2)
    idx = flat.argmax(dim=-1)
    h, w = hm.shape[-2:]
    x = (idx % w).float() / max(w - 1, 1)
    y = (idx // w).float() / max(h - 1, 1)
    return idx, torch.stack([x, y], dim=-1)


def _radius_mass(prob, gt_idx, radius_px, h, w):
    yy, xx = torch.meshgrid(
        torch.arange(h, device=prob.device),
        torch.arange(w, device=prob.device),
        indexing="ij",
    )
    gx = (gt_idx % w).view(*gt_idx.shape, 1, 1)
    gy = (gt_idx // w).view(*gt_idx.shape, 1, 1)
    keep = (xx.view(1, 1, h, w) - gx).square() + (yy.view(1, 1, h, w) - gy).square()
    keep = keep <= float(radius_px) ** 2
    return (prob.view(*gt_idx.shape, h, w) * keep.float()).flatten(2).sum(-1)


def _radius_keep(gt_idx, radius_px, h, w):
    yy, xx = torch.meshgrid(
        torch.arange(h, device=gt_idx.device),
        torch.arange(w, device=gt_idx.device),
        indexing="ij",
    )
    gx = (gt_idx % w).view(*gt_idx.shape, 1, 1)
    gy = (gt_idx // w).view(*gt_idx.shape, 1, 1)
    dist2 = (xx.view(1, 1, h, w) - gx).square() + (yy.view(1, 1, h, w) - gy).square()
    return dist2 <= float(radius_px) ** 2


def _gaussian_target(gt_idx, radius_px, sigma_px, h, w, dtype):
    yy, xx = torch.meshgrid(
        torch.arange(h, device=gt_idx.device),
        torch.arange(w, device=gt_idx.device),
        indexing="ij",
    )
    gx = (gt_idx % w).view(*gt_idx.shape, 1, 1)
    gy = (gt_idx // w).view(*gt_idx.shape, 1, 1)
    dist2 = (xx.view(1, 1, h, w) - gx).square() + (yy.view(1, 1, h, w) - gy).square()
    keep = dist2 <= float(radius_px) ** 2
    target = torch.exp(-dist2.to(dtype) / (2.0 * float(sigma_px) ** 2)) * keep.to(dtype)
    return target / (target.flatten(2).sum(-1).view(*gt_idx.shape, 1, 1) + 1e-8)


def peak_mass_losses(pred, base, gt, mask, video_names, args):
    pred_k = pred[:, :-1].float()
    base_k = base[:, :-1].float()
    gt_k = gt[:, :-1].float()
    mask_k = mask[:, :-1].float()
    b, c, h, w = pred_k.shape
    flat_pred = pred_k.flatten(2)
    flat_base = base_k.flatten(2)
    gt_idx, gt_xy = peak_xy_from_heatmap(gt_k)
    gt_present = mask_k > 0.5

    prob = torch.softmax(flat_pred * args.peak_beta, dim=-1)
    mass = _radius_mass(prob, gt_idx, args.mass_radius_px, h, w).clamp_min(1e-8)
    tight_mass = _radius_mass(prob, gt_idx, args.tight_radius_px, h, w).clamp_min(1e-8)
    base_at_gt = flat_base.gather(2, gt_idx.unsqueeze(-1)).squeeze(-1)
    hard_missing = gt_present & (base_at_gt < args.missing_base_thresh)
    missing_weight = gt_present.float() * (1.0 + args.missing_hard_extra * hard_missing.float())
    nll = -mass.log() / np.log(h * w)
    missing = (nll * missing_weight).sum() / (missing_weight.sum() + 1e-6)

    tight_nll = -tight_mass.log() / np.log(h * w)
    loc = (tight_nll * gt_present.float()).sum() / (gt_present.float().sum() + 1e-6)

    out_mass = 1.0 - _radius_mass(prob, gt_idx, args.outlier_dist_px, h, w)
    outlier = (out_mass * gt_present.float()).sum() / (gt_present.float().sum() + 1e-6)

    smooth_terms = []
    pred_xy = softargmax_xy(pred)
    for i in range(1, b):
        if video_names[i] != video_names[i - 1]:
            continue
        present = gt_present[i] & gt_present[i - 1]
        if present.any():
            pred_delta = pred_xy[i] - pred_xy[i - 1]
            gt_delta = gt_xy[i] - gt_xy[i - 1]
            smooth_terms.append(torch.nn.functional.smooth_l1_loss(pred_delta[present], gt_delta[present]))
    peak_smooth = torch.stack(smooth_terms).mean() if smooth_terms else pred.new_zeros(())
    present_count = gt_present.float().sum() + 1e-6
    mass_mean = (mass * gt_present.float()).sum() / present_count
    tight_mass_mean = (tight_mass * gt_present.float()).sum() / present_count
    return missing, loc, outlier, peak_smooth, mass_mean, tight_mass_mean


def peak_sharp_losses(pred, base, gt, mask, video_names, args):
    pred_k = pred[:, :-1].float()
    base_k = base[:, :-1].float()
    gt_k = gt[:, :-1].float()
    mask_k = mask[:, :-1].float()
    b, c, h, w = pred_k.shape
    flat_pred = pred_k.flatten(2)
    gt_idx, gt_xy = peak_xy_from_heatmap(gt_k)
    gt_present = mask_k > 0.5

    logits = flat_pred * args.peak_beta
    logp = torch.log_softmax(logits, dim=-1)
    prob = logp.exp()
    target = _gaussian_target(
        gt_idx, args.mass_radius_px, args.peak_sigma_px, h, w, pred_k.dtype
    ).flatten(2)
    sharp = -(target * logp).sum(-1) / np.log(h * w)
    sharp = (sharp * gt_present.float()).sum() / (gt_present.float().sum() + 1e-6)

    inside = _radius_keep(gt_idx, args.tight_radius_px, h, w).flatten(2)
    outside = ~_radius_keep(gt_idx, args.mass_radius_px, h, w).flatten(2)
    inside_peak = logits.masked_fill(~inside, -1e4).max(-1).values
    outside_peak = logits.masked_fill(~outside, -1e4).max(-1).values
    rank = torch.nn.functional.softplus(outside_peak - inside_peak + args.peak_rank_margin)
    rank = (rank * gt_present.float()).sum() / (gt_present.float().sum() + 1e-6)

    mass = _radius_mass(prob, gt_idx, args.mass_radius_px, h, w).clamp_min(1e-8)
    tight_mass = _radius_mass(prob, gt_idx, args.tight_radius_px, h, w).clamp_min(1e-8)
    out_mass = 1.0 - _radius_mass(prob, gt_idx, args.outlier_dist_px, h, w)
    outlier = (out_mass * gt_present.float()).sum() / (gt_present.float().sum() + 1e-6)
    teacher = (((pred_k - base_k).square()) * mask_k[:, :, None, None]).sum()
    teacher = teacher / (mask_k.sum() * h * w + 1e-6)

    smooth_terms = []
    pred_xy = softargmax_xy(pred)
    for i in range(1, b):
        if video_names[i] != video_names[i - 1]:
            continue
        present = gt_present[i] & gt_present[i - 1]
        if present.any():
            pred_delta = pred_xy[i] - pred_xy[i - 1]
            gt_delta = gt_xy[i] - gt_xy[i - 1]
            smooth_terms.append(torch.nn.functional.smooth_l1_loss(pred_delta[present], gt_delta[present]))
    peak_smooth = torch.stack(smooth_terms).mean() if smooth_terms else pred.new_zeros(())
    present_count = gt_present.float().sum() + 1e-6
    mass_mean = (mass * gt_present.float()).sum() / present_count
    tight_mass_mean = (tight_mass * gt_present.float()).sum() / present_count
    return sharp, rank, outlier, peak_smooth, mass_mean, tight_mass_mean, teacher


def train_batch(pending, model, hrnet, optimizer, scaler, device, use_amp, args, step, totals):
    windows = torch.stack([item[0] for item in pending]).to(device)
    gt = torch.stack([item[1] for item in pending]).to(device)
    mask = torch.stack([item[2] for item in pending]).to(device)
    video_names = [item[3] for item in pending]
    center = windows[:, -1]

    with torch.no_grad():
        base = hrnet(center).float()
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        pred = model(windows)
        heat = masked_mse(pred.float(), gt, mask, args.fg_weight)
        cont = continuity_loss(pred.float(), video_names)
        missing = loc = outlier = peak_smooth = mass = tight_mass = teacher = pred.new_zeros(())
        if args.loss_mode in {"peak_hard", "peak_mass"}:
            missing, loc, outlier, peak_smooth, mass, tight_mass = peak_mass_losses(
                pred.float(), base, gt, mask, video_names, args
            )
            loss = (
                args.heat_weight * heat
                + args.missing_weight * missing
                + args.location_weight * loc
                + args.outlier_weight * outlier
                + args.peak_smooth_weight * peak_smooth
                + args.continuity_weight * cont
            )
        elif args.loss_mode == "peak_sharp":
            missing, loc, outlier, peak_smooth, mass, tight_mass, teacher = peak_sharp_losses(
                pred.float(), base, gt, mask, video_names, args
            )
            loss = (
                args.heat_weight * heat
                + args.missing_weight * missing
                + args.location_weight * loc
                + args.outlier_weight * outlier
                + args.teacher_weight * teacher
                + args.peak_smooth_weight * peak_smooth
                + args.continuity_weight * cont
            )
        else:
            loss = heat + args.continuity_weight * cont

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    step += 1
    totals["loss"] += loss.item()
    totals["heat"] += heat.item()
    totals["cont"] += cont.item()
    totals["missing"] += missing.item()
    totals["loc"] += loc.item()
    totals["outlier"] += outlier.item()
    totals["peak_smooth"] += peak_smooth.item()
    totals["mass"] += mass.item()
    totals["tight_mass"] += tight_mass.item()
    totals["teacher"] += teacher.item()
    totals["fg_base"] += fg_mse(base, gt, mask)
    totals["fg_fused"] += fg_mse(pred.float(), gt, mask)
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
    ap.add_argument("--fusion-level", choices=["first", "last"], required=True)
    ap.add_argument("--window-size", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--fg-weight", type=float, default=50.0)
    ap.add_argument("--continuity-weight", type=float, default=0.01)
    ap.add_argument("--loss-mode", choices=["mse", "peak_hard", "peak_mass", "peak_sharp"], default="mse")
    ap.add_argument("--heat-weight", type=float, default=0.05)
    ap.add_argument("--missing-weight", type=float, default=8.0)
    ap.add_argument("--location-weight", type=float, default=8.0)
    ap.add_argument("--outlier-weight", type=float, default=0.5)
    ap.add_argument("--peak-smooth-weight", type=float, default=0.005)
    ap.add_argument("--peak-beta", type=float, default=10.0)
    ap.add_argument("--mass-radius-px", type=float, default=5.0)
    ap.add_argument("--tight-radius-px", type=float, default=1.0)
    ap.add_argument("--peak-sigma-px", type=float, default=1.0)
    ap.add_argument("--peak-rank-margin", type=float, default=1.0)
    ap.add_argument("--teacher-weight", type=float, default=1.0)
    ap.add_argument("--missing-base-thresh", type=float, default=0.05)
    ap.add_argument("--missing-hard-extra", type=float, default=3.0)
    ap.add_argument("--outlier-dist-px", type=float, default=12.0)
    ap.add_argument("--residual-scale", type=float, default=1.0)
    ap.add_argument("--head-lora-rank", type=int, default=0)
    ap.add_argument("--head-lora-scale", type=float, default=0.05)
    ap.add_argument("--peak-thresh", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260628)
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
        head_lora_rank=args.head_lora_rank,
        head_lora_scale=args.head_lora_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    resize = T.Resize((540, 960))
    to_tensor = T.ToTensor()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    def checkpoint_payload(epoch, steps):
        return {
            "state_dict": model.adapter_state_dict(),
            "model": "TemporalHRNetFeatureFusion",
            "fusion_level": args.fusion_level,
            "window_size": args.window_size,
            "continuity_weight": args.continuity_weight,
            "loss_mode": args.loss_mode,
            "loss_weights": {
                "heat": args.heat_weight,
                "missing": args.missing_weight,
                "location": args.location_weight,
                "outlier": args.outlier_weight,
                "peak_smooth": args.peak_smooth_weight,
                "teacher": args.teacher_weight,
            },
            "mass_radius_px": args.mass_radius_px,
            "tight_radius_px": args.tight_radius_px,
            "peak_sigma_px": args.peak_sigma_px,
            "peak_rank_margin": args.peak_rank_margin,
            "residual_scale": args.residual_scale,
            "head_lora_rank": args.head_lora_rank,
            "head_lora_scale": args.head_lora_scale,
            "split": args.split,
            "videos": video_names,
            "epoch": epoch,
            "steps": steps,
        }

    print(
        f"feature-fusion split={args.split} level={args.fusion_level} videos={video_names} "
        f"window={args.window_size} batch={args.batch_size} cont={args.continuity_weight} loss={args.loss_mode}",
        flush=True,
    )

    global_step = 0
    run_start = time.perf_counter()
    use_amp = device.type == "cuda"
    for epoch in range(args.epochs):
        random.shuffle(video_names)
        pending = []
        totals = {
            "loss": 0.0, "heat": 0.0, "cont": 0.0,
            "missing": 0.0, "loc": 0.0, "outlier": 0.0, "peak_smooth": 0.0,
            "mass": 0.0, "tight_mass": 0.0, "teacher": 0.0,
            "fg_base": 0.0, "fg_fused": 0.0, "batches": 0, "frames": 0,
        }
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
                    global_step = train_batch(
                        pending, model, hrnet, optimizer, scaler, device, use_amp,
                        args, global_step, totals,
                    )
                    if args.log_every and global_step != before and global_step % args.log_every == 0:
                        elapsed = time.perf_counter() - run_start
                        print(
                            f"epoch={epoch} step={global_step} "
                            f"loss={totals['loss']/max(totals['batches'],1):.6f} "
                            f"heat={totals['heat']/max(totals['batches'],1):.6f} "
                            f"miss={totals['missing']/max(totals['batches'],1):.4f} "
                            f"loc={totals['loc']/max(totals['batches'],1):.4f} "
                            f"mass={totals['mass']/max(totals['batches'],1):.4f} "
                            f"tight={totals['tight_mass']/max(totals['batches'],1):.4f} "
                            f"out={totals['outlier']/max(totals['batches'],1):.3e} "
                            f"teacher={totals['teacher']/max(totals['batches'],1):.3e} "
                            f"psm={totals['peak_smooth']/max(totals['batches'],1):.3e} "
                            f"cont={totals['cont']/max(totals['batches'],1):.3e} "
                            f"fg={totals['fg_base']/max(totals['batches'],1):.6f}->"
                            f"{totals['fg_fused']/max(totals['batches'],1):.6f} "
                            f"frames/s={totals['frames']/max(elapsed,1):.2f}",
                            flush=True,
                        )
                if args.max_steps and global_step >= args.max_steps:
                    break
            if pending:
                global_step = train_batch(
                    pending, model, hrnet, optimizer, scaler, device, use_amp,
                    args, global_step, totals,
                )
            if args.max_steps and global_step >= args.max_steps:
                break
        epoch_path = args.out.replace(".pt", f"_epoch{epoch + 1}.pt")
        torch.save(checkpoint_payload(epoch, global_step), epoch_path)
        torch.save(checkpoint_payload(epoch, global_step), args.out)
        print(
            f"epoch={epoch} frames={totals['frames']} "
            f"loss={totals['loss']/max(totals['batches'],1):.6f} "
            f"heat={totals['heat']/max(totals['batches'],1):.6f} "
            f"miss={totals['missing']/max(totals['batches'],1):.4f} "
            f"loc={totals['loc']/max(totals['batches'],1):.4f} "
            f"mass={totals['mass']/max(totals['batches'],1):.4f} "
            f"tight={totals['tight_mass']/max(totals['batches'],1):.4f} "
            f"out={totals['outlier']/max(totals['batches'],1):.3e} "
            f"teacher={totals['teacher']/max(totals['batches'],1):.3e} "
            f"psm={totals['peak_smooth']/max(totals['batches'],1):.3e} "
            f"cont={totals['cont']/max(totals['batches'],1):.3e} "
            f"fg={totals['fg_base']/max(totals['batches'],1):.6f}->"
            f"{totals['fg_fused']/max(totals['batches'],1):.6f} saved={epoch_path}",
            flush=True,
        )
        if args.max_steps and global_step >= args.max_steps:
            break
    torch.save(checkpoint_payload(epoch, global_step), args.out)
    print(f"saved final -> {args.out}; steps={global_step}")


if __name__ == "__main__":
    main()
