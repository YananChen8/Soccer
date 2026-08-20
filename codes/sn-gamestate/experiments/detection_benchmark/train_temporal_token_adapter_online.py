"""Online SNGS2024 training for sparse-token temporal adapters."""
import argparse
import random
import time
from collections import deque
from pathlib import Path

import torch

from sn_gamestate.temporal_hrnet import KeypointTokenTemporalAdapter, heatmaps_to_tokens
from train_temporal_adapter_online import (
    load_frame,
    load_hrnet,
    masked_mse,
    read_videos,
)
import torchvision.transforms as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--architecture", choices=["tcn", "stgcn", "transformer"], required=True)
    ap.add_argument("--window-size", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--hrnet-batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam-res", type=float, default=0.01)
    ap.add_argument("--lam-distill", type=float, default=0.1)
    ap.add_argument("--fg-weight", type=float, default=50.0)
    ap.add_argument("--peak-thresh", type=float, default=0.1)
    ap.add_argument("--residual-scale", type=float, default=1.0)
    ap.add_argument("--max-shift-px", type=float, default=12.0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260624)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    all_videos = read_videos(
        "/remote-home/jiayuanrao/yishan/datasets/SoccerNetGS_2024_nbjw", args.split
    )
    missing = [video for video in args.videos if video not in all_videos]
    if missing:
        raise ValueError(f"missing videos: {missing}")

    hrnet = load_hrnet(device)
    adapter = KeypointTokenTemporalAdapter(
        channels=58,
        window_size=args.window_size,
        architecture=args.architecture,
        hidden=args.hidden,
        residual_scale=args.residual_scale,
        max_shift_px=args.max_shift_px,
    ).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    resize, to_tensor = T.Resize((540, 960)), T.ToTensor()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    windows_per_epoch = sum(
        max(0, len(all_videos[video]) - args.window_size + 1) for video in args.videos
    )
    print(
        f"architecture={args.architecture} window={args.window_size} "
        f"videos={len(args.videos)} frames={sum(len(all_videos[v]) for v in args.videos)} "
        f"windows/epoch={windows_per_epoch}"
    )

    step = 0
    start = time.perf_counter()
    video_names = list(args.videos)
    for epoch in range(args.epochs):
        random.shuffle(video_names)
        epoch_start = time.perf_counter()
        epoch_steps = 0
        for video in video_names:
            history = deque(maxlen=args.window_size)
            frames = all_videos[video]
            for chunk_start in range(0, len(frames), args.hrnet_batch_size):
                chunk = frames[chunk_start:chunk_start + args.hrnet_batch_size]
                loaded = [
                    load_frame(stem, resize, to_tensor, args.peak_thresh)
                    for _image_id, stem in chunk
                ]
                images = torch.stack([item[0] for item in loaded]).to(device)
                with torch.no_grad():
                    heatmaps = hrnet(images).detach()
                    tokens = heatmaps_to_tokens(heatmaps).detach()
                del images

                for index in range(len(chunk)):
                    history.append(tokens[index])
                    if len(history) < args.window_size:
                        continue
                    token_window = torch.stack(tuple(history), dim=0).unsqueeze(0)
                    current = heatmaps[index:index + 1]
                    gt = loaded[index][1].unsqueeze(0).to(device)
                    mask = loaded[index][2].unsqueeze(0).to(device)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        refined, delta = adapter(token_window, current)
                        hm_loss = masked_mse(refined.float(), gt, mask, args.fg_weight)
                        res_loss = delta.float().square().mean()
                        distill = (refined.float() - current.float()).square().mean()
                        loss = hm_loss + args.lam_res * res_loss + args.lam_distill * distill
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    step += 1
                    epoch_steps += 1
                    if step % args.log_every == 0:
                        elapsed = time.perf_counter() - start
                        print(
                            f"epoch={epoch} video={video} step={step} loss={loss.item():.6f} "
                            f"windows/s={step/elapsed:.3f} "
                            f"gpu_mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB"
                        )
                del heatmaps, tokens, loaded

        seconds = time.perf_counter() - epoch_start
        print(f"epoch={epoch} windows={epoch_steps} seconds={seconds:.1f}")

    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "which": "kp_token",
            "channels": 58,
            "window_size": args.window_size,
            "architecture": args.architecture,
            "hidden": args.hidden,
            "residual_scale": args.residual_scale,
            "max_shift_px": args.max_shift_px,
            "videos": args.videos,
            "steps": step,
        },
        args.out,
    )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
