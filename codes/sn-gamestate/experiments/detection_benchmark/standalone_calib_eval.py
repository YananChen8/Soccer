"""Standalone field-calibration eval for temporal adapters on TEST 116/117/118.

NO tracklab. Reuses cached frozen-HRNet test heatmaps + the existing line/point
accuracy metric (structured_calibration.metrics.accuracy_eval). For each adapter
(+baseline) we replicate nbjw's exact decode+solve to produce per-frame camera
`parameters`, then score line meanAccuracy / precision / recall / reprojection.

Per-adapter result is printed AND appended to --out as soon as it finishes, so
partial results survive an early stop.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from nbjw_calib.utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    complete_keypoints, coords_to_dict)
from nbjw_calib.utils.utils_calib import FramebyFrameCalib
from sn_gamestate.temporal_hrnet import (
    TemporalHeatmapAdapter, KeypointTokenTemporalAdapter,
    SparseTemporalKeypointAdapter, heatmaps_to_tokens, pad_window)
from sn_gamestate.structured_calibration.metrics import accuracy_eval

SNG = "/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate"
CACHE = f"{SNG}/outputs/gsr/temporal_hrnet/heatmap_cache_full_20260624/sngs2024_test"
DATA_ROOT = f"{SNG}/datasets/SoccerNetGS/test"
CKPT = f"{SNG}/outputs/gsr/temporal_hrnet"
VIDS = ["116", "117", "118"]  # metric GT loader prepends "SNGS-"

# name -> checkpoint path (None = baseline, no adapter)
ADAPTERS = {
    "baseline":        None,
    "v1_dense_gen_k3": f"{CKPT}/ckpt/kp_adapter_gen.pt",
    "3dcnn_k15":       f"{CKPT}/quick_subset12/3dcnn_k15/kp_adapter_3dcnn_k15.pt",
    "online_w15":      f"{CKPT}/online_w15_sngs2024/kp_adapter_online_w15.pt",
    "token_tcn_k50":   f"{CKPT}/quick_subset12/tcn_k50/kp_adapter_tcn_k50.pt",
    "token_stgcn_k50": f"{CKPT}/quick_subset12/stgcn_k50/kp_adapter_stgcn_k50.pt",
    "token_transformer_k50": f"{CKPT}/quick_subset12/transformer_k50/kp_adapter_transformer_k50.pt",
}


def load_adapter(path, device):
    """Build the right adapter class from the checkpoint; return (callable, K).
    callable(h_seq[1,K,58,h,w]) -> refined[1,58,h,w]."""
    if path is None:
        return None, 1
    ck = torch.load(path, map_location=device)
    which = ck.get("which", "")
    if ck.get("model_family") == "sparse_keypoint":
        a = SparseTemporalKeypointAdapter(
            ck["channels"], ck["window_size"], architecture=ck["architecture"],
            hidden=ck.get("hidden", 64), residual_scale=ck.get("residual_scale", 1.0),
            max_shift_px=ck.get("max_shift_px", 8.0))
        a.load_state_dict(ck["state_dict"]); a.to(device).eval()
        return (lambda win: a(win)[0]), ck["window_size"]
    if which == "kp_token":
        a = KeypointTokenTemporalAdapter(
            channels=ck["channels"], window_size=ck["window_size"],
            architecture=ck["architecture"], hidden=ck.get("hidden", 64),
            residual_scale=ck.get("residual_scale", 1.0),
            max_shift_px=ck.get("max_shift_px", 12.0))
        a.load_state_dict(ck["state_dict"]); a.to(device).eval()

        def fwd(win):  # win [1,K,C,h,w] -> tokens [1,K,C,3]
            toks = torch.stack([heatmaps_to_tokens(win[:, t]) for t in range(win.shape[1])], dim=1)
            return a(toks, win[:, -1])[0]
        return fwd, ck["window_size"]
    # dense (which == "kp" or default)
    a = TemporalHeatmapAdapter(
        ck["channels"], ck["window_size"], residual_scale=ck.get("residual_scale", 1.0),
        adapter_type=ck.get("adapter_type", "depthwise_conv3d"),
        mix_hidden=ck.get("mix_hidden", 128))
    a.load_state_dict(ck["state_dict"]); a.to(device).eval()
    return (lambda win: a(win)[0]), ck["window_size"]


def solve_params_for_video(vid, fwd, K, device, stride=1):
    files = sorted(glob.glob(f"{CACHE}/SNGS-{vid}/frame_*.npz"))
    buf, out = [], {}
    cam = FramebyFrameCalib(1920, 1080, denormalize=True)
    for i, f in enumerate(files):
        d = np.load(f)
        kp = torch.from_numpy(d["kp_hm"].astype(np.float32)).unsqueeze(0).to(device)
        frame = int(d["frame"])
        if fwd is not None:
            buf.append(kp)              # accumulate EVERY frame for window continuity
            if len(buf) > K:
                buf.pop(0)
        if stride > 1 and i % stride != 0:
            continue                    # only decode+solve frames the metric scores
        line = torch.from_numpy(d["line_hm"].astype(np.float32)).unsqueeze(0).to(device)
        if fwd is not None:
            with torch.no_grad():
                refined = fwd(pad_window(torch.stack(buf, dim=1), K))
        else:
            refined = kp
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(refined[:, :-1])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(line[:, :-1])
        kp_dict = coords_to_dict(kp_coords, threshold=0.1449)
        lines_dict = coords_to_dict(line_coords, threshold=0.2983)
        pred = complete_keypoints(kp_dict, lines_dict, w=960, h=540, normalize=True)[0]
        cam.update(pred)
        h = cam.get_homography_from_ground_plane(use_ransac=50, inverse=True)
        params = {}
        if h is not None:
            try:
                vr = cam.heuristic_voting()
                params = vr["cam_params"] if vr else {}
            except Exception:
                params = {}
        out[str(frame)] = params
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10, help="accuracy subsample (per-frame mean is robust)")
    ap.add_argument("--out", default=f"{CKPT}/analysis/standalone_test_116_118.json")
    ap.add_argument("--only", nargs="*", default=None, help="subset of adapter names")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    names = args.only or list(ADAPTERS)
    results = {}
    if os.path.isfile(args.out):
        results = json.load(open(args.out))
    for name in names:
        if name in results:
            print(f"[skip] {name} already done"); continue
        t0 = time.time()
        fwd, K = load_adapter(ADAPTERS[name], args.device)
        params_by_vid = {v: solve_params_for_video(v, fwd, K, args.device, args.stride) for v in VIDS}
        res = accuracy_eval(params_by_vid, DATA_ROOT, VIDS, nproc=3, stride=args.stride)
        res["window"] = K
        results[name] = res
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"[{name}] K={K} meanAcc={res.get('meanAccuracy')} "
              f"completeness={res.get('completeness')} "
              f"meanReproj={res.get('meanReproj')} ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE -> {args.out}")


if __name__ == "__main__":
    main()
