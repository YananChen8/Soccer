import argparse
import copy
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

import tmp_official_aux_report_eval_visual_20260701 as ref
import tmp_true_tta_nbjw_20260702 as tta


def peak_xy(hm):
    # hm: C,H,W
    c, h, w = hm.shape
    flat = hm.reshape(c, -1)
    idx = flat.argmax(axis=1)
    y = idx // w
    x = idx % w
    val = flat[np.arange(c), idx]
    return x.astype(float), y.astype(float), val.astype(float)


def channel_world_x(nc):
    xs = []
    for ch in range(nc):
        if ch < len(tta.keypoint_world_coords_2D):
            xs.append(float(tta.keypoint_world_coords_2D[ch][0]))
        else:
            xs.append(52.5)
    return xs


def aligned_flip_output(model, img, swap):
    flip_img = torch.flip(img, dims=[3])
    y = model(flip_img)
    y = torch.flip(y, dims=[3])
    return y[:, swap, :, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["116", "117", "118", "119", "120", "121", "122", "123"])
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    frames_root, _ = ref.get_split_paths("test")
    kp_swap = tta.keypoint_swap()
    kp_model, _ = ref.base.load_hrnets(device)
    kp_model.eval()
    base_state = copy.deepcopy(kp_model.state_dict())
    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    tta_args = types.SimpleNamespace(
        steps=1, lr=1e-5, anchor_weight=0.05, peak_weight=0.0,
        pseudo_conf_threshold=0.05, pseudo_ransac_px=20.0,
        pseudo_sigma=2.0, outlier_weight=0.001,
        temporal_sigma=2.0, temporal_gate_px=30.0, temporal_weight=0.2,
    )

    rows = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    for video in [str(v).replace("SNGS-", "") for v in args.videos]:
        video_rows = []
        files = sorted((frames_root / f"SNGS-{video}" / "img1").glob("*.jpg"))
        for idx, path in enumerate(files):
            if idx % args.stride:
                continue
            pil = Image.open(path).convert("RGB").resize((960, 540))
            img = tfm(pil).unsqueeze(0).to(device)

            kp_model.load_state_dict(base_state)
            kp_model.eval()
            with torch.no_grad():
                raw = kp_model(img)
                raw_flip = aligned_flip_output(kp_model, img, kp_swap)

            kp_model.load_state_dict(base_state)
            tta.adapt_model(kp_model, img, kp_swap, "flip_consistency", tta_args)
            kp_model.eval()
            with torch.no_grad():
                aft = kp_model(img)
                aft_flip = aligned_flip_output(kp_model, img, kp_swap)

            raw_np = raw[0].detach().cpu().numpy()
            raw_flip_np = raw_flip[0].detach().cpu().numpy()
            aft_np = aft[0].detach().cpu().numpy()
            aft_flip_np = aft_flip[0].detach().cpu().numpy()
            nc = raw_np.shape[0]
            wx = channel_world_x(nc)
            rx, ry, rv = peak_xy(raw_np)
            rfx, rfy, rfv = peak_xy(raw_flip_np)
            ax, ay, av = peak_xy(aft_np)
            afx, afy, afv = peak_xy(aft_flip_np)
            for ch in range(nc):
                video_rows.append({
                    "video": video,
                    "frame": path.stem,
                    "channel": ch,
                    "world_x": wx[ch],
                    "before_peak_dist": float(np.hypot(rx[ch] - rfx[ch], ry[ch] - rfy[ch])),
                    "after_peak_dist": float(np.hypot(ax[ch] - afx[ch], ay[ch] - afy[ch])),
                    "before_abs_conf_gap": float(abs(rv[ch] - rfv[ch])),
                    "after_abs_conf_gap": float(abs(av[ch] - afv[ch])),
                    "before_mse": float(np.mean((raw_np[ch] - raw_flip_np[ch]) ** 2)),
                    "after_mse": float(np.mean((aft_np[ch] - aft_flip_np[ch]) ** 2)),
                })
        rows.extend(video_rows)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"done video={video} rows={len(video_rows)} total={len(rows)}", flush=True)
    print(out, flush=True)


if __name__ == "__main__":
    main()
