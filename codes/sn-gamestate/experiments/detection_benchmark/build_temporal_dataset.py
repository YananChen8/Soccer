"""Build the temporal-adapter training set from SoccerNetGS_2024_nbjw or
SoccerNetCalibration_2023_nbjw.

For each frame of each requested video, emits one npz containing BOTH the frozen
HRNet prediction heatmaps AND the GT heatmaps (rendered from the line-GT JSON),
all on the SAME 270x480 grid:

    kp_hm    float16 [58,270,480]  frozen-HRNet kp  output  (pre [:, :-1])
    line_hm  float16 [24,270,480]  frozen-HRNet line output
    kp_gt    float16 [58,270,480]  GT kp heatmap   (KeypointsDB)
    kp_mask  float16 [58]          per-channel valid mask
    line_gt  float16 [24,270,480]  GT line heatmap (LineKeypointsDB)
    frame    int                   frame number (from image_id)

For SoccerNetGS_2024, frames are grouped per video and ordered by image_id, so
HeatmapWindowDataset can build temporal windows WITHIN a video (never across videos).
For SoccerNetCalibration_2023, each frame is cached as a singleton pseudo-video
(`CAL23_<stem>`). That gives dense GT supervision without inventing fake temporal
continuity across unrelated frames.

Alignment fix (important): GS images are full-res 1920x1080. We resize to 540x960
*before both* HRNet and GT render, so HRNet out and GT are both 270x480 (=÷2).
GT line coords are normalized, so the resize needs no separate coord transform.

Run on 202 (GPU):
  PY=/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python
  cd .../sn-gamestate
  PYTHONPATH=plugins/calibration:. CUDA_VISIBLE_DEVICES=0 \
    $PY experiments/detection_benchmark/build_temporal_dataset.py \
      --split valid --videos SNGS-021 SNGS-034 --max-frames 0
"""
import argparse
import csv
import json
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
import torchvision.transforms as T

from nbjw_calib.model.cls_hrnet import get_cls_net
from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
from nbjw_calib.utils.utils_keypoints import KeypointsDB
from nbjw_calib.utils.utils_lines import LineKeypointsDB

DATASET_SNGS2024 = "/remote-home/jiayuanrao/yishan/datasets/SoccerNetGS_2024_nbjw"
DATASET_CAL2023 = "/remote-home/jiayuanrao/yishan/datasets/SoccerNetCalibration_2023_nbjw"
CFG = "sn_gamestate/configs/modules/pitch/nbjw_calib.yaml"
CKPT_DIR = "/remote-home/jiayuanrao/yishan/sn-gamestate/pretrained_models/calibration"
DEFAULT_OUT_ROOT = "outputs/gsr/temporal_hrnet/heatmap_cache"


def load_models(device):
    cfg = yaml.safe_load(open(CFG))
    mk = get_cls_net(cfg["cfg"])
    mk.load_state_dict(torch.load(f"{CKPT_DIR}/SV_kp", map_location=device))
    ml = get_cls_net_l(cfg["cfg_l"])
    ml.load_state_dict(torch.load(f"{CKPT_DIR}/SV_lines", map_location=device))
    return mk.to(device).eval(), ml.to(device).eval()


def read_manifest_sngs2024(dataset_root, split, videos):
    """Return {video: [(image_id, stem_path), ...sorted by frame]}."""
    by_video = defaultdict(list)
    want = set(videos) if videos else None
    with open(f"{dataset_root}/{split}_manifest.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if want and row["video"] not in want:
                continue
            stem = f"{dataset_root}/{split}/{row['image_id']}"
            by_video[row["video"]].append((int(row["image_id"]), stem))
    for v in by_video:
        by_video[v].sort()
    return by_video


def read_manifest_cal2023(dataset_root, split, max_samples, start_sample=0):
    """Return singleton pseudo-videos for dense calibration GT."""
    split_root = Path(dataset_root) / split
    items = []
    for json_path in sorted(split_root.glob("*.json"))[start_sample:]:
        stem = json_path.stem
        jpg_path = split_root / f"{stem}.jpg"
        if not jpg_path.exists():
            continue
        items.append((f"CAL23_{stem}", [(int(stem), str(split_root / stem))]))
        if max_samples and len(items) >= max_samples:
            break
    return dict(items)


def prepare_frame(video, stem, image_id, out_dir, resize, to_tensor, skip_existing=False):
    out_path = out_dir / f"frame_{image_id:010d}.npz"
    if skip_existing and out_path.exists():
        return None
    img = Image.open(stem + ".jpg").convert("RGB")
    img = resize(img)                       # 540x960
    x = to_tensor(img)                      # [3,540,960]
    data = json.load(open(stem + ".json"))
    # KeypointsDB has a known goal-label key fix.
    if "Goal left post left" in data:
        data["Goal left post left "] = data.pop("Goal left post left")
    try:
        kp_gt, kp_mask = KeypointsDB(data, x).get_tensor_w_mask()  # (58,270,480),(58,)
    except Exception:
        print(f"WARNING: invalid keypoint GT for {stem}; using an all-zero masked target")
        traceback.print_exc()
        kp_gt = np.zeros((58, 270, 480), dtype=np.float32)
        kp_mask = np.zeros(58, dtype=np.float32)
    try:
        line_gt = LineKeypointsDB(data, x).get_tensor()            # (24,270,480)
    except Exception:
        print(f"WARNING: invalid line GT for {stem}; using an all-zero target")
        traceback.print_exc()
        line_gt = np.zeros((24, 270, 480), dtype=np.float32)
    return video, out_path, image_id, x, kp_gt, kp_mask, line_gt


def cache_batch(items, mk, ml, device):
    """Run both frozen HRNets once for a batch, then save one npz per frame."""
    if not items:
        return []
    xb = torch.stack([item[3] for item in items]).to(device)
    with torch.no_grad():
        kp_hm = mk(xb).float().cpu().numpy().astype(np.float16)
        line_hm = ml(xb).float().cpu().numpy().astype(np.float16)
    for i, (_video, out_path, image_id, _x, kp_gt, kp_mask, line_gt) in enumerate(items):
        np.savez_compressed(
            out_path,
            kp_hm=kp_hm[i], line_hm=line_hm[i],
            kp_gt=kp_gt.astype(np.float16), kp_mask=kp_mask.astype(np.float16),
            line_gt=line_gt.astype(np.float16), frame=image_id,
        )
    return [item[0] for item in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-type", choices=["sngs2024", "cal2023"], default="sngs2024")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out-split", default=None, help="cache split name; default=same as --split")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--videos", nargs="*", default=None, help="e.g. SNGS-021; default=all in split")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames per video")
    ap.add_argument("--max-samples", type=int, default=0, help="only for cal2023 singleton caching")
    ap.add_argument("--start-sample", type=int, default=0, help="only for cal2023 chunked caching")
    ap.add_argument("--batch-size", type=int, default=1, help="frames per frozen-HRNet forward pass")
    ap.add_argument("--skip-existing", action="store_true", help="do not recompute existing frame_*.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    mk, ml = load_models(args.device)
    resize = T.Resize((540, 960))
    to_tensor = T.ToTensor()

    dataset_root = args.dataset_root
    if dataset_root is None:
        dataset_root = DATASET_SNGS2024 if args.dataset_type == "sngs2024" else DATASET_CAL2023
    out_split = args.out_split or args.split
    if args.dataset_type == "sngs2024":
        by_video = read_manifest_sngs2024(dataset_root, args.split, args.videos)
    else:
        by_video = read_manifest_cal2023(dataset_root, args.split, args.max_samples, args.start_sample)
    print(f"{len(by_video)} videos: {list(by_video)}")

    cached_by_video = defaultdict(int)
    frame_count_by_video = {}
    pending = []

    def flush():
        for cached_video in cache_batch(pending, mk, ml, args.device):
            cached_by_video[cached_video] += 1
        pending.clear()

    for video, frames in by_video.items():
        if args.dataset_type == "sngs2024" and args.max_frames:
            frames = frames[: args.max_frames]
        frame_count_by_video[video] = len(frames)
        out_dir = Path(args.out_root) / out_split / video
        out_dir.mkdir(parents=True, exist_ok=True)
        for image_id, stem in frames:
            item = prepare_frame(video, stem, image_id, out_dir, resize, to_tensor, args.skip_existing)
            if item is None:
                continue
            pending.append(item)
            if len(pending) >= args.batch_size:
                flush()
    flush()
    for video in by_video:
        out_dir = Path(args.out_root) / out_split / video
        print(f"[{video}] cached {cached_by_video[video]}/{frame_count_by_video[video]} frames -> {out_dir}")
    print("done.")


if __name__ == "__main__":
    main()
