"""Fastest full-test calib eval for the 3 TOKEN adapters (+baseline).

Two decoupled stages so GPU work and CPU scoring each parallelize fully:

  mode=params  (GPU, run one per free GPU over a video chunk):
      live HRNet + adapter + solve -> per-frame camera `parameters`.
      Dumps params_by_imgid_per_video for ALL adapters to a json. NO scoring.
  mode=score   (CPU, single launch, nproc=many):
      load all param jsons, score line meanAcc (macro) + point acc (micro) +
      reproj with a multiprocessing pool over (adapter, video) jobs.

This removes the per-frame inline get_polylines (the old bottleneck) from the
GPU loop, and parallelizes scoring across all cores. Expected: params ~3 min on
6 GPUs (or ~18 min on 1), score ~8 min.

Stage 1 (per GPU):
  CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 PYTHONPATH=plugins/calibration:.:experiments/detection_benchmark \
    python fast_full_test.py --mode params --videos 116 117 ... --out PARAMS/g7.json
Stage 2 (once, after all params done):
    python fast_full_test.py --mode score --params-glob 'PARAMS/g*.json' --out RESULT.json --nproc 32
"""
import argparse
import glob
import json
from multiprocessing import Pool

import numpy as np
import torch

SNG = "/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate"
DATA_ROOT = f"{SNG}/datasets/SoccerNetGS/test"
ADAPTER_NAMES = ["baseline", "token_tcn_k50", "token_stgcn_k50", "token_transformer_k50"]


# ---------- stage 1: params (GPU) ----------
def run_params(videos, out, stride, device):
    from pathlib import Path
    import torchvision.transforms as T
    from PIL import Image
    import yaml
    from nbjw_calib.model.cls_hrnet import get_cls_net
    from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
    from nbjw_calib.utils.utils_heatmap import (
        get_keypoints_from_heatmap_batch_maxpool,
        get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, coords_to_dict)
    from nbjw_calib.utils.utils_calib import FramebyFrameCalib
    from sn_gamestate.temporal_hrnet import pad_window
    from standalone_calib_eval import load_adapter, ADAPTERS as A
    cfg = yaml.safe_load(open(f"{SNG}/sn_gamestate/configs/modules/pitch/nbjw_calib.yaml"))
    cd = "/remote-home/jiayuanrao/yishan/sn-gamestate/pretrained_models/calibration"
    mk = get_cls_net(cfg["cfg"]); mk.load_state_dict(torch.load(f"{cd}/SV_kp", map_location=device)); mk.to(device).eval()
    ml = get_cls_net_l(cfg["cfg_l"]); ml.load_state_dict(torch.load(f"{cd}/SV_lines", map_location=device)); ml.to(device).eval()
    adapters = {n: load_adapter(A[n], device) for n in ADAPTER_NAMES}
    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    out_data = {}
    for vid in videos:
        files = sorted(glob.glob(f"{DATA_ROOT}/SNGS-{vid}/img1/*.jpg"))
        stems = [Path(f).stem for f in files]
        kp_seq, line_at = [], {}
        for i, f in enumerate(files):
            x = tfm(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                kp_seq.append(mk(x).half())
                if i % stride == 0:
                    line_at[i] = ml(x).half()
        out_data[vid] = {}
        for name in ADAPTER_NAMES:
            fwd, K = adapters[name]
            cam = FramebyFrameCalib(1920, 1080, denormalize=True)
            buf, pb = [], {}
            for i in range(len(kp_seq)):
                kp = kp_seq[i].float()
                if fwd is not None:
                    buf.append(kp)
                    if len(buf) > K:
                        buf.pop(0)
                if i % stride != 0 or i not in line_at:
                    continue
                refined = fwd(pad_window(torch.stack(buf, 1), K)) if fwd is not None else kp
                kc = get_keypoints_from_heatmap_batch_maxpool(refined[:, :-1])
                lc = get_keypoints_from_heatmap_batch_maxpool_l(line_at[i].float()[:, :-1])
                pred = complete_keypoints(coords_to_dict(kc, threshold=0.1449),
                                          coords_to_dict(lc, threshold=0.2983), w=960, h=540, normalize=True)[0]
                cam.update(pred)
                h = cam.get_homography_from_ground_plane(use_ransac=50, inverse=True)
                params = {}
                if h is not None:
                    try:
                        vr = cam.heuristic_voting(); params = vr["cam_params"] if vr else {}
                    except Exception:
                        params = {}
                pb[f"3{vid}{stems[i]}"] = params
            out_data[vid][name] = pb
        json.dump(out_data, open(out, "w"))
        print(f"[params {vid}] done", flush=True)
    print(f"PARAMS_DONE -> {out}")


# ---------- stage 2: score (CPU, parallel) ----------
def _score_job(args):
    vid, adapter, pb = args
    from sn_gamestate.structured_calibration.metrics import (
        get_polylines, evaluate_camera_prediction, mirror_labels,
        load_gt_lines_for_video, THRESHOLD, WIDTH, HEIGHT)
    gt = load_gt_lines_for_video(DATA_ROOT, vid)
    micro, macro, reproj = [], [], []
    for gid, params in pb.items():
        if gid not in gt or not isinstance(params, dict) or not params:
            continue
        try:
            pred = get_polylines(params, WIDTH, HEIGHT, sampling_factor=0.9)
        except Exception:
            continue

        def both(gl):
            c, _, r = evaluate_camera_prediction(pred, gl, THRESHOLD)
            mi = c[0, 0] / c.sum() if c.sum() > 0 else 0.0
            pl = [np.mean([e <= THRESHOLD for e in es]) for es in r.values() if es]
            ma = float(np.mean(pl)) if pl else 0.0
            rj = [e for es in r.values() for e in es]
            return mi, ma, rj
        m1 = both(gt[gid])
        try:
            m2 = both(mirror_labels(gt[gid]))
        except Exception:
            m2 = None
        m = m1 if (m2 is None or m1[0] >= m2[0]) else m2
        micro.append(m[0]); macro.append(m[1]); reproj += m[2]
    return adapter, vid, micro, macro, reproj


def run_score(params_glob, out, nproc):
    params = {}
    for f in glob.glob(params_glob):
        for vid, ad in json.load(open(f)).items():
            params.setdefault(vid, {}).update(ad)
    jobs = [(vid, ad, params[vid][ad]) for vid in params for ad in ADAPTER_NAMES if ad in params[vid]]
    with Pool(nproc) as p:
        parts = p.map(_score_job, jobs)
    agg = {a: {"micro": [], "macro": [], "reproj": [], "nvid": 0} for a in ADAPTER_NAMES}
    for ad, vid, mi, ma, rj in parts:
        agg[ad]["micro"] += mi; agg[ad]["macro"] += ma; agg[ad]["reproj"] += rj; agg[ad]["nvid"] += 1
    res = {}
    for a in ADAPTER_NAMES:
        g = agg[a]
        res[a] = {"point_acc": float(np.mean(g["micro"])) if g["micro"] else None,
                  "line_acc": float(np.mean(g["macro"])) if g["macro"] else None,
                  "reproj_mean": float(np.mean(g["reproj"])) if g["reproj"] else None,
                  "n_frames": len(g["micro"]), "n_videos": g["nvid"]}
    json.dump(res, open(out, "w"), indent=2)
    base = res["baseline"]["point_acc"]
    for a in ADAPTER_NAMES:
        r = res[a]; d = "" if a == "baseline" else f"  (Δ {r['point_acc']-base:+.4f})"
        print(f"{a:24s} point={r['point_acc']:.4f} line={r['line_acc']:.4f} reproj={r['reproj_mean']:.2f}{d}")
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["params", "score"], required=True)
    ap.add_argument("--videos", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--params-glob")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.mode == "params":
        run_params(a.videos, a.out, a.stride, a.device if torch.cuda.is_available() else "cpu")
    else:
        run_score(a.params_glob, a.out, a.nproc)


if __name__ == "__main__":
    main()
