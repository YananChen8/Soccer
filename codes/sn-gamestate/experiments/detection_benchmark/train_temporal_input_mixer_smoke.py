"""Smoke-train an input-level temporal mixer before frozen NBJW HRNet.

This is intentionally small: train only a K-frame RGB mixer, keep HRNet frozen,
and supervise the resulting keypoint heatmap with existing NBJW GT heatmaps.

No command in this file is launched automatically. Use --init-only for a
baseline-identical checkpoint, or --max-steps for a bounded smoke.
"""
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
from sn_gamestate.temporal_hrnet import TemporalInputMixer


DATASET_ROOT = "/remote-home/jiayuanrao/yishan/datasets/SoccerNetGS_2024_nbjw"
CFG = "sn_gamestate/configs/modules/pitch/nbjw_calib.yaml"
CKPT = "/remote-home/jiayuanrao/yishan/sn-gamestate/pretrained_models/calibration/SV_kp"


def read_videos(dataset_root, split):
    videos = defaultdict(list)
    manifest = Path(dataset_root) / f"{split}_manifest.tsv"
    with manifest.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            image_id = int(row["image_id"])
            stem = Path(dataset_root) / split / row["image_id"]
            videos[row["video"]].append((image_id, stem))
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
    pad = [frames[0]] * (window_size - len(frames))
    return torch.stack(pad + list(frames))


def masked_mse(pred, gt, mask, fg_weight):
    channel_mask = mask[:, :, None, None]
    weight = channel_mask * (1.0 + fg_weight * (gt > 0.1).float())
    return ((pred - gt).square() * weight).sum() / (weight.sum() + 1e-6)


def fg_mse(pred, gt, mask):
    fg = (gt > 0.1).float() * mask[:, :, None, None]
    return (((pred - gt).square() * fg).sum() / (fg.sum() + 1e-6)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DATASET_ROOT)
    ap.add_argument("--split", default="train")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--window-size", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--fg-weight", type=float, default=50.0)
    ap.add_argument("--peak-thresh", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--init-only", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260627)
    ap.add_argument("--out", default="outputs/gsr/temporal_hrnet/temporal_calib_results_hub/input_mixer_smoke/checkpoints/input_mixer_k3.pt")
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
        video_names = video_names[:args.max_videos]

    mixer = TemporalInputMixer(args.window_size).to(device)
    optimizer = torch.optim.Adam(mixer.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    resize = T.Resize((540, 960))
    to_tensor = T.ToTensor()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    def checkpoint_payload(epoch, steps):
        return {
            "state_dict": mixer.state_dict(),
            "model": "TemporalInputMixer",
            "window_size": args.window_size,
            "dataset": "SoccerNetGS_2024_nbjw",
            "split": args.split,
            "videos": video_names,
            "epoch": epoch,
            "steps": steps,
        }

    if args.init_only:
        torch.save(checkpoint_payload(-1, 0), args.out)
        print(f"saved identity input mixer -> {args.out}")
        return

    hrnet = load_hrnet(device)
    print(
        f"input-mixer smoke split={args.split} videos={video_names} "
        f"window={args.window_size} batch={args.batch_size}",
        flush=True,
    )

    global_step = 0
    run_start = time.perf_counter()
    use_amp = device.type == "cuda"

    for epoch in range(args.epochs):
        random.shuffle(video_names)
        pending = []
        totals = {"loss": 0.0, "fg_base": 0.0, "fg_mix": 0.0, "batches": 0, "frames": 0}
        for video in video_names:
            history = []
            frames = videos[video]
            if args.max_frames:
                frames = frames[: args.max_frames]
            for _image_id, stem in frames:
                image, gt, mask = load_frame(stem, resize, to_tensor, args.peak_thresh)
                history.append(image)
                if len(history) > args.window_size:
                    history.pop(0)
                window = left_pad_window(history, args.window_size)
                pending.append((window, gt, mask))
                if len(pending) >= args.batch_size:
                    before_step = global_step
                    global_step = train_batch(
                        pending, mixer, hrnet, optimizer, scaler, device, use_amp,
                        args.fg_weight, global_step, totals,
                    )
                    if args.log_every and global_step != before_step and global_step % args.log_every == 0:
                        elapsed = time.perf_counter() - run_start
                        print(
                            f"epoch={epoch} step={global_step} "
                            f"loss={totals['loss']/max(totals['batches'],1):.6f} "
                            f"fg={totals['fg_base']/max(totals['batches'],1):.6f}->"
                            f"{totals['fg_mix']/max(totals['batches'],1):.6f} "
                            f"frames/s={totals['frames']/max(elapsed,1):.2f}",
                            flush=True,
                        )
                if args.max_steps and global_step >= args.max_steps:
                    break
            if args.max_steps and global_step >= args.max_steps:
                break
        if pending and not (args.max_steps and global_step >= args.max_steps):
            global_step = train_batch(
                pending, mixer, hrnet, optimizer, scaler, device, use_amp,
                args.fg_weight, global_step, totals,
            )
        epoch_path = args.out.replace(".pt", f"_epoch{epoch + 1}.pt")
        torch.save(checkpoint_payload(epoch, global_step), epoch_path)
        torch.save(checkpoint_payload(epoch, global_step), args.out)
        print(
            f"epoch={epoch} frames={totals['frames']} "
            f"loss={totals['loss']/max(totals['batches'],1):.6f} "
            f"fg={totals['fg_base']/max(totals['batches'],1):.6f}->"
            f"{totals['fg_mix']/max(totals['batches'],1):.6f} "
            f"saved={epoch_path}",
            flush=True,
        )
        if args.max_steps and global_step >= args.max_steps:
            break
    torch.save(checkpoint_payload(epoch, global_step), args.out)
    print(f"saved final -> {args.out}; steps={global_step}")


def train_batch(pending, mixer, hrnet, optimizer, scaler, device, use_amp, fg_weight, step, totals):
    windows = torch.stack([item[0] for item in pending]).to(device)
    gt = torch.stack([item[1] for item in pending]).to(device)
    mask = torch.stack([item[2] for item in pending]).to(device)
    center = windows[:, -1]
    with torch.no_grad():
        base = hrnet(center).float()
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        mixed = mixer(windows)
        pred = hrnet(mixed)
        loss = masked_mse(pred.float(), gt, mask, fg_weight)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    step += 1
    totals["loss"] += loss.item()
    totals["fg_base"] += fg_mse(base, gt, mask)
    totals["fg_mix"] += fg_mse(pred.float(), gt, mask)
    totals["batches"] += 1
    totals["frames"] += len(pending)
    pending.clear()
    return step


if __name__ == "__main__":
    main()
