"""Full-test-set standalone calib eval for the 3 TOKEN adapters (+baseline).

Live HRNet (no cache, no disk): for each video, run the frozen HRNet once over
all frames (kept in RAM), then replay each adapter on that heatmap sequence ->
decode -> solve -> camera params -> score. Outputs, per adapter:
  point_acc (micro, official confusion)   + line_acc (macro, per-line averaged)
  completeness, reproj mean/median, and ADAPTER-ONLY latency (ms/frame).

Parallelize by splitting --videos across GPUs (one process each), then merge.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from nbjw_calib.model.cls_hrnet import get_cls_net
from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
from nbjw_calib.utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, coords_to_dict)
from nbjw_calib.utils.utils_calib import FramebyFrameCalib
from sn_gamestate.temporal_hrnet import pad_window
from sn_gamestate.structured_calibration.metrics import (
    get_polylines, evaluate_camera_prediction, mirror_labels,
    load_gt_lines_for_video, THRESHOLD, WIDTH, HEIGHT)
from standalone_calib_eval import load_adapter, ADAPTERS as _ALL  # reuse loader

SNG = "/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate"
FRAMES = f"{SNG}/datasets/SoccerNetGS/test"
DATA_ROOT = f"{SNG}/datasets/SoccerNetGS/test"
TOKEN = {k: _ALL[k] for k in ["baseline", "token_tcn_k50", "token_stgcn_k50", "token_transformer_k50"]}
CFG = f"{SNG}/sn_gamestate/configs/modules/pitch/nbjw_calib.yaml"
CKPT_DIR = "/remote-home/jiayuanrao/yishan/sn-gamestate/pretrained_models/calibration"


def load_hrnet(device):
    import yaml
    cfg = yaml.safe_load(open(CFG))
    mk = get_cls_net(cfg["cfg"]); mk.load_state_dict(torch.load(f"{CKPT_DIR}/SV_kp", map_location=device))
    ml = get_cls_net_l(cfg["cfg_l"]); ml.load_state_dict(torch.load(f"{CKPT_DIR}/SV_lines", map_location=device))
    return mk.to(device).eval(), ml.to(device).eval()


def score_frame(params, gt_lines):
    """Return (point_acc_micro, line_acc_macro, reproj_list) or None if no cam."""
    if not isinstance(params, dict) or not params:
        return None
    try:
        pred = get_polylines(params, WIDTH, HEIGHT, sampling_factor=0.9)
    except Exception:
        return None

    def both(gl):
        c, _, r = evaluate_camera_prediction(pred, gl, THRESHOLD)
        micro = c[0, 0] / c.sum() if c.sum() > 0 else 0.0
        per_line = [np.mean([e <= THRESHOLD for e in errs]) for errs in r.values() if errs]
        macro = float(np.mean(per_line)) if per_line else 0.0
        reproj = [e for errs in r.values() for e in errs]
        return micro, macro, reproj

    m1 = both(gt_lines)
    try:
        m2 = both(mirror_labels(gt_lines))
    except Exception:
        m2 = None
    return m1 if (m2 is None or m1[0] >= m2[0]) else m2


def eval_video(vid, hrnet_kp, hrnet_l, adapters, device, stride):
    files = sorted(glob.glob(f"{FRAMES}/SNGS-{vid}/img1/*.jpg"))
    stems = [Path(f).stem for f in files]   # 6-digit frame numbers
    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    gt = load_gt_lines_for_video(DATA_ROOT, vid)
    # test image_id convention: "3" + 3-digit video + 6-digit frame (matches cache)
    # 1) HRNet once over all frames -> RAM (fp16)
    kp_seq, line_at = [], {}
    frame_ids = []
    for i, f in enumerate(files):
        # frame id from filename stem matched to GT order by index
        x = tfm(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            hk = hrnet_kp(x)
        kp_seq.append(hk.half())
        if stride <= 1 or i % stride == 0:
            with torch.no_grad():
                line_at[i] = hrnet_l(x).half()
        frame_ids.append(i)
    out = {}
    for name in adapters:
        fwd, K = adapters[name]
        cam = FramebyFrameCalib(1920, 1080, denormalize=True)
        buf = []
        micro, macro, reproj, n, t_ad = [], [], [], 0, 0.0
        for i in range(len(kp_seq)):
            kp = kp_seq[i].float()
            if fwd is not None:
                buf.append(kp)
                if len(buf) > K:
                    buf.pop(0)
            if stride > 1 and i % stride != 0:
                continue
            if i not in line_at:
                continue
            if fwd is not None:
                win = pad_window(torch.stack(buf, dim=1), K)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    refined = fwd(win)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t_ad += time.perf_counter() - t0
            else:
                refined = kp
            kc = get_keypoints_from_heatmap_batch_maxpool(refined[:, :-1])
            lc = get_keypoints_from_heatmap_batch_maxpool_l(line_at[i].float()[:, :-1])
            pred = complete_keypoints(coords_to_dict(kc, threshold=0.1449),
                                      coords_to_dict(lc, threshold=0.2983),
                                      w=960, h=540, normalize=True)[0]
            cam.update(pred)
            h = cam.get_homography_from_ground_plane(use_ransac=50, inverse=True)
            params = {}
            if h is not None:
                try:
                    vr = cam.heuristic_voting(); params = vr["cam_params"] if vr else {}
                except Exception:
                    params = {}
            gid = f"3{vid}{stems[i]}"          # test id = 3+video+frame
            sc = score_frame(params, gt[gid]) if gid in gt else None
            n += 1
            if sc is not None:
                micro.append(sc[0]); macro.append(sc[1]); reproj += sc[2]
        nf = max(1, sum(1 for i in range(len(kp_seq)) if not (stride > 1 and i % stride != 0)))
        out[name] = {
            "point_acc": float(np.mean(micro)) if micro else None,
            "line_acc": float(np.mean(macro)) if macro else None,
            "n_scored": len(micro), "n_total": n,
            "completeness": len(micro) / n if n else None,
            "reproj_mean": float(np.mean(reproj)) if reproj else None,
            "reproj_median": float(np.median(reproj)) if reproj else None,
            "adapter_ms_per_frame": (t_ad / n * 1000.0) if (fwd is not None and n) else 0.0,
            "window": K,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    hrnet_kp, hrnet_l = load_hrnet(dev)
    adapters = {name: load_adapter(TOKEN[name], dev) for name in TOKEN}
    results = {}
    for vid in args.videos:
        t0 = time.time()
        results[vid] = eval_video(vid, hrnet_kp, hrnet_l, adapters, dev, args.stride)
        json.dump(results, open(args.out, "w"), indent=2)
        bl = results[vid]["baseline"]["point_acc"]
        st = results[vid]["token_stgcn_k50"]["point_acc"]
        print(f"[{vid}] base_pt={bl:.3f} stgcn_pt={st:.3f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE_CHUNK -> {args.out}")


if __name__ == "__main__":
    main()
