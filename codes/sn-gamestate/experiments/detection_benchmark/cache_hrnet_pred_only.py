"""Cache frozen HRNet prediction heatmaps only.

Used for test/inference splits where GT heatmaps are not needed. Output npz keys
match the subset consumed by cached_full_test_round2.py: kp_hm, line_hm, frame.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image

from nbjw_calib.model.cls_hrnet import get_cls_net
from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l


DATASET_SNGS2024 = "/remote-home/jiayuanrao/yishan/datasets/SoccerNetGS_2024_nbjw"
CFG = "sn_gamestate/configs/modules/pitch/nbjw_calib.yaml"
CKPT_DIR = "/remote-home/jiayuanrao/yishan/sn-gamestate/pretrained_models/calibration"


def load_models(device):
    cfg = yaml.safe_load(open(CFG))
    mk = get_cls_net(cfg["cfg"])
    mk.load_state_dict(torch.load(f"{CKPT_DIR}/SV_kp", map_location=device))
    ml = get_cls_net_l(cfg["cfg_l"])
    ml.load_state_dict(torch.load(f"{CKPT_DIR}/SV_lines", map_location=device))
    return mk.to(device).eval(), ml.to(device).eval()


def read_manifest(dataset_root, split, videos):
    by_video = defaultdict(list)
    want = set(videos) if videos else None
    with open(f"{dataset_root}/{split}_manifest.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if want and row["video"] not in want:
                continue
            stem = f"{dataset_root}/{split}/{row['image_id']}"
            by_video[row["video"]].append((int(row["image_id"]), stem))
    for video in by_video:
        by_video[video].sort()
    return by_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DATASET_SNGS2024)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-split", default=None)
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    mk, ml = load_models(args.device)
    resize = T.Resize((540, 960))
    to_tensor = T.ToTensor()
    by_video = read_manifest(args.dataset_root, args.split, args.videos)
    out_split = args.out_split or args.split
    print(f"{len(by_video)} videos: {list(by_video)}", flush=True)

    for video, frames in by_video.items():
        out_dir = Path(args.out_root) / out_split / video
        out_dir.mkdir(parents=True, exist_ok=True)
        pending = []
        for image_id, stem in frames:
            out_path = out_dir / f"frame_{image_id:010d}.npz"
            if args.skip_existing and out_path.exists():
                continue
            x = to_tensor(resize(Image.open(stem + ".jpg").convert("RGB")))
            pending.append((image_id, out_path, x))
            if len(pending) >= args.batch_size:
                flush(pending, mk, ml, args.device)
        flush(pending, mk, ml, args.device)
        n = len(list(out_dir.glob("frame_*.npz")))
        print(f"[{video}] cached_total {n}/{len(frames)} frames -> {out_dir}", flush=True)
    print("done.", flush=True)


def flush(pending, mk, ml, device):
    if not pending:
        return
    xb = torch.stack([item[2] for item in pending]).to(device)
    with torch.no_grad():
        kp_hm = mk(xb).float().cpu().numpy().astype(np.float16)
        line_hm = ml(xb).float().cpu().numpy().astype(np.float16)
    for i, (image_id, out_path, _x) in enumerate(pending):
        np.savez_compressed(out_path, kp_hm=kp_hm[i], line_hm=line_hm[i], frame=image_id)
    pending.clear()


if __name__ == "__main__":
    main()
