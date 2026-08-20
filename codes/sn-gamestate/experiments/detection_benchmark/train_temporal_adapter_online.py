"""Online training for the keypoint temporal adapter on SoccerNetGS 2024.

The frozen HRNet runs once per frame per epoch. Videos are shuffled, frames
remain ordered, and a rolling heatmap buffer forms fixed-length windows.
No heatmap cache is read or written.
"""
import argparse
import csv
import json
import random
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image

from nbjw_calib.model.cls_hrnet import get_cls_net
from nbjw_calib.utils.utils_keypoints import KeypointsDB
from sn_gamestate.temporal_hrnet import TemporalHeatmapAdapter


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
        print(f"WARNING: invalid keypoint GT for {stem}; masking this frame")
        traceback.print_exc()
        gt = np.zeros((58, 270, 480), dtype=np.float32)
        mask = np.zeros(58, dtype=np.float32)
    return x, torch.from_numpy(gt), torch.from_numpy(mask)


def masked_mse(pred, gt, mask, fg_weight):
    channel_mask = mask[:, :, None, None]
    weight = channel_mask * (1.0 + fg_weight * (gt > 0.1).float())
    return ((pred - gt).square() * weight).sum() / (weight.sum() + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DATASET_ROOT)
    ap.add_argument("--split", default="train")
    ap.add_argument("--window-size", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--hrnet-batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam-res", type=float, default=0.01)
    ap.add_argument("--lam-distill", type=float, default=0.1)
    ap.add_argument("--fg-weight", type=float, default=50.0)
    ap.add_argument("--peak-thresh", type=float, default=0.1)
    ap.add_argument("--residual-scale", type=float, default=1.0)
    ap.add_argument("--mix-hidden", type=int, default=128)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--videos", nargs="*", default=None, help="explicit complete videos to use")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--init-only", action="store_true", help="save zero-init epoch-0 checkpoint and exit")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260624)
    ap.add_argument("--out", default="outputs/gsr/temporal_hrnet/ckpt/kp_adapter_online_w15.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.window_size != 15:
        raise ValueError("This experiment fixes --window-size=15")

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

    hrnet = load_hrnet(device)
    adapter = TemporalHeatmapAdapter(
        channels=58,
        window_size=15,
        residual_scale=args.residual_scale,
        adapter_type="depthwise_full_window",
        mix_hidden=args.mix_hidden,
    ).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    resize = T.Resize((540, 960))
    to_tensor = T.ToTensor()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    total_frames = sum(len(videos[name]) for name in video_names)
    expected_windows = sum(max(0, len(videos[name]) - 14) for name in video_names)
    print(
        f"online SNGS2024 split={args.split} videos={len(video_names)} "
        f"frames={total_frames} windows/epoch={expected_windows} window=15 "
        f"hrnet_batch={args.hrnet_batch_size}"
    )

    global_step = 0
    run_start = time.perf_counter()

    def checkpoint_payload(epoch):
        return {
            "state_dict": adapter.state_dict(),
            "which": "kp",
            "channels": 58,
            "window_size": 15,
            "adapter_type": "depthwise_full_window",
            "mix_hidden": args.mix_hidden,
            "residual_scale": args.residual_scale,
            "dataset": "SoccerNetGS_2024_nbjw",
            "split": args.split,
            "videos": video_names,
            "epoch": epoch,
            "steps": global_step,
        }

    if args.init_only:
        torch.save(checkpoint_payload(-1), args.out)
        print(f"saved zero-init epoch 0 -> {args.out}")
        return

    last_epoch = -1
    for epoch in range(args.epochs):
        last_epoch = epoch
        random.shuffle(video_names)
        epoch_start = time.perf_counter()
        epoch_windows = 0
        for video in video_names:
            history = deque(maxlen=15)
            frames = videos[video]
            for chunk_start in range(0, len(frames), args.hrnet_batch_size):
                chunk = frames[chunk_start:chunk_start + args.hrnet_batch_size]
                loaded = [
                    load_frame(stem, resize, to_tensor, args.peak_thresh)
                    for _image_id, stem in chunk
                ]
                images = torch.stack([item[0] for item in loaded]).to(device)
                with torch.inference_mode():
                    heatmaps = hrnet(images).detach()
                del images

                for index, (_image_id, _stem) in enumerate(chunk):
                    history.append(heatmaps[index])
                    if len(history) < 15:
                        continue
                    window = torch.stack(tuple(history), dim=0).unsqueeze(0)
                    gt = loaded[index][1].unsqueeze(0).to(device)
                    mask = loaded[index][2].unsqueeze(0).to(device)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        refined, delta = adapter(window)
                        loss_hm = masked_mse(refined.float(), gt, mask, args.fg_weight)
                        loss_res = delta.float().square().mean()
                        loss_distill = (refined.float() - window[:, -1].float()).square().mean()
                        loss = loss_hm + args.lam_res * loss_res + args.lam_distill * loss_distill
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    global_step += 1
                    epoch_windows += 1

                    if global_step % args.log_every == 0:
                        elapsed = time.perf_counter() - run_start
                        print(
                            f"epoch={epoch} video={video} step={global_step} "
                            f"loss={loss.item():.6f} windows/s={global_step/elapsed:.3f} "
                            f"gpu_mem={torch.cuda.max_memory_allocated(device)/2**30:.1f}GiB"
                        )
                    if args.max_steps and global_step >= args.max_steps:
                        break
                del heatmaps, loaded
                if args.max_steps and global_step >= args.max_steps:
                    break
            if args.max_steps and global_step >= args.max_steps:
                break

        epoch_seconds = time.perf_counter() - epoch_start
        print(
            f"epoch={epoch} windows={epoch_windows} seconds={epoch_seconds:.1f} "
            f"windows/s={epoch_windows/max(epoch_seconds, 1e-6):.3f}"
        )
        epoch_path = args.out.replace(".pt", f"_epoch{epoch + 1}.pt")
        torch.save(checkpoint_payload(epoch), epoch_path)
        torch.save(checkpoint_payload(epoch), args.out)
        print(f"saved epoch -> {epoch_path}; latest -> {args.out}")
        if args.max_steps and global_step >= args.max_steps:
            break

    elapsed = time.perf_counter() - run_start
    torch.save(checkpoint_payload(last_epoch), args.out)
    print(f"saved -> {args.out}; steps={global_step}; elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
