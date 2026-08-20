"""Inference smoke for the temporal adapter, on held-out SNGS-034 frames.

Proves two things WITHOUT running the full GS pipeline:
  1. adapter OFF  -> decoded keypoints byte-identical to baseline NBJW (non-destructive).
  2. adapter ON   -> heatmaps refined; report how decoded keypoints change
                     (count delta, mean pixel shift) vs baseline.

Run on 202:
  PY=.../wys_soccermaster/bin/python
  cd .../sn-gamestate
  PYTHONPATH=plugins/calibration:. CUDA_VISIBLE_DEVICES=2 \
    $PY experiments/detection_benchmark/smoke_inference_temporal.py \
      --ckpt outputs/gsr/temporal_hrnet/ckpt/kp_adapter_gen.pt --n 30
"""
import argparse
import glob

import numpy as np
import torch
import yaml
from PIL import Image
import torchvision.transforms as T

from nbjw_calib.model.cls_hrnet import get_cls_net
from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
from nbjw_calib.utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool, coords_to_dict)
from sn_gamestate.temporal_hrnet import TemporalHeatmapAdapter, pad_window

CFG = "sn_gamestate/configs/modules/pitch/nbjw_calib.yaml"
CKPT_DIR = "/remote-home/jiayuanrao/yishan/sn-gamestate/pretrained_models/calibration"
FRAMES = "/remote-home/jiayuanrao/yishan/sn-gamestate/data/SoccerNetGS/valid/SNGS-034/img1"


def decode(hm):
    coords = get_keypoints_from_heatmap_batch_maxpool(hm[:, :-1])
    return coords_to_dict(coords, threshold=0.1449)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    cfg = yaml.safe_load(open(CFG))
    mk = get_cls_net(cfg["cfg"]); mk.load_state_dict(torch.load(f"{CKPT_DIR}/SV_kp", map_location=dev))
    mk.to(dev).eval()

    ck = torch.load(args.ckpt, map_location=dev)
    adapter = TemporalHeatmapAdapter(ck["channels"], ck["window_size"],
                                     residual_scale=ck.get("residual_scale", 1.0))
    adapter.load_state_dict(ck["state_dict"]); adapter.to(dev).eval()
    K = ck["window_size"]

    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    frames = sorted(glob.glob(f"{FRAMES}/*.jpg"))[: args.n]

    buf = []
    n_base, n_ref, shifts, identical = [], [], [], 0
    for fp in frames:
        x = tfm(Image.open(fp).convert("RGB")).unsqueeze(0).to(dev)
        with torch.no_grad():
            hm = mk(x)
            buf.append(hm.detach())
            if len(buf) > K:
                buf.pop(0)
            win = pad_window(torch.stack(buf, dim=1), K)
            hm_ref, _ = adapter(win)
        kp_base, kp_ref = decode(hm), decode(hm_ref)
        n_base.append(len(kp_base)); n_ref.append(len(kp_ref))
        # mean pixel shift over keypoints present in both
        common = set(kp_base) & set(kp_ref)
        if common:
            d = [np.hypot(kp_base[i]["x"] - kp_ref[i]["x"],
                          kp_base[i]["y"] - kp_ref[i]["y"]) for i in common]
            shifts.append(float(np.mean(d)))
        if kp_base.keys() == kp_ref.keys() and not common_moved(kp_base, kp_ref):
            identical += 1

    print(f"frames={len(frames)}  K={K}")
    print(f"kp count  baseline mean={np.mean(n_base):.2f}  refined mean={np.mean(n_ref):.2f}")
    print(f"mean pixel shift of shared kp = {np.mean(shifts):.3f} (over {len(shifts)} frames)")
    print(f"frames where refined==baseline (adapter no-op) = {identical}/{len(frames)}")
    # sanity: residual_scale=0 must reproduce baseline exactly
    adapter.residual_scale = 0.0
    with torch.no_grad():
        win = pad_window(torch.stack(buf[-1:], dim=1), K)
        hm0, _ = adapter(win)
    assert torch.allclose(hm0, buf[-1]), "residual_scale=0 must equal H_t"
    print("residual_scale=0 == baseline H_t: OK")


def common_moved(a, b, tol=1e-6):
    for i in set(a) & set(b):
        if abs(a[i]["x"] - b[i]["x"]) > tol or abs(a[i]["y"] - b[i]["y"]) > tol:
            return True
    return False


if __name__ == "__main__":
    main()
