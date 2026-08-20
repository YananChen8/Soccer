"""Evaluate TemporalInputMixer against frozen HRNet baseline on SoccerNetGS test.

Runs the keypoint HRNet twice on sampled frames: once on the original current
frame and once on the K-frame input-mixer output. The line HRNet remains the
same original-frame line prediction for both paths, isolating keypoint changes.
"""
import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image

from nbjw_calib.model.cls_hrnet import get_cls_net
from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
from nbjw_calib.utils.utils_calib import FramebyFrameCalib
from nbjw_calib.utils.utils_heatmap import (
    complete_keypoints,
    coords_to_dict,
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
)
from sn_gamestate.temporal_hrnet import TemporalInputMixer
from sn_gamestate.structured_calibration.metrics import (
    HEIGHT,
    THRESHOLD,
    WIDTH,
    evaluate_camera_prediction,
    get_polylines,
    load_gt_lines_for_video,
    mirror_labels,
)

SNG = "/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate"
FRAMES = f"{SNG}/datasets/SoccerNetGS/test"
DATA_ROOT = f"{SNG}/datasets/SoccerNetGS/test"
CFG = f"{SNG}/sn_gamestate/configs/modules/pitch/nbjw_calib.yaml"
CKPT_DIR = "/remote-home/jiayuanrao/yishan/sn-gamestate/pretrained_models/calibration"
KP_DECODE_THRESHOLD = float(os.environ.get("NBJW_KP_THRESHOLD", "0.1449"))
LINE_DECODE_THRESHOLD = float(os.environ.get("NBJW_LINE_THRESHOLD", "0.1"))


def load_hrnets(device):
    cfg = yaml.safe_load(open(CFG))
    kp = get_cls_net(cfg["cfg"])
    kp.load_state_dict(torch.load(f"{CKPT_DIR}/SV_kp", map_location=device))
    line = get_cls_net_l(cfg["cfg_l"])
    line.load_state_dict(torch.load(f"{CKPT_DIR}/SV_lines", map_location=device))
    for model in (kp, line):
        model.to(device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
    return kp, line


def load_mixer(path, device):
    ck = torch.load(path, map_location=device)
    mixer = TemporalInputMixer(window_size=int(ck["window_size"]))
    mixer.load_state_dict(ck["state_dict"])
    mixer.to(device).eval()
    return mixer, int(ck["window_size"]), ck


def left_pad_window(frames, window_size):
    if len(frames) >= window_size:
        return torch.stack(frames[-window_size:])
    pad = [frames[0]] * (window_size - len(frames))
    return torch.stack(pad + list(frames))


def decode_keypoints(kp_hm, line_hm):
    kp = get_keypoints_from_heatmap_batch_maxpool(kp_hm[:, :-1])
    line = get_keypoints_from_heatmap_batch_maxpool_l(line_hm[:, :-1])
    return complete_keypoints(
        coords_to_dict(kp, threshold=KP_DECODE_THRESHOLD),
        coords_to_dict(line, threshold=LINE_DECODE_THRESHOLD),
        w=960,
        h=540,
        normalize=True,
    )[0]


def solve_params(keypoints):
    cam = FramebyFrameCalib(1920, 1080, denormalize=True)
    cam.update(keypoints)
    h = cam.get_homography_from_ground_plane(use_ransac=50, inverse=True)
    if h is None:
        return {}
    try:
        voted = cam.heuristic_voting()
        return voted["cam_params"] if voted else {}
    except Exception:
        return {}


def score_frame(params, gt_lines):
    if not isinstance(params, dict) or not params:
        return None
    try:
        pred = get_polylines(params, WIDTH, HEIGHT, sampling_factor=0.9)
    except Exception:
        return None

    def one_side(lines):
        c, _, r = evaluate_camera_prediction(pred, lines, THRESHOLD)
        point = c[0, 0] / c.sum() if c.sum() > 0 else 0.0
        per_line = [np.mean([err <= THRESHOLD for err in errs]) for errs in r.values() if errs]
        line_acc = float(np.mean(per_line)) if per_line else 0.0
        reproj = [err for errs in r.values() for err in errs]
        return point, line_acc, reproj

    base = one_side(gt_lines)
    try:
        mirrored = one_side(mirror_labels(gt_lines))
    except Exception:
        mirrored = None
    return base if mirrored is None or base[0] >= mirrored[0] else mirrored


def summarize(items):
    point = [x["point_acc"] for x in items if x["point_acc"] is not None]
    line = [x["line_acc"] for x in items if x["line_acc"] is not None]
    reproj = [v for x in items for v in x["reproj"]]
    return {
        "point_acc": float(np.mean(point)) if point else None,
        "line_acc": float(np.mean(line)) if line else None,
        "reproj_mean": float(np.mean(reproj)) if reproj else None,
        "reproj_median": float(np.median(reproj)) if reproj else None,
        "n_scored": len(point),
        "n_total": len(items),
        "completeness": len(point) / len(items) if items else None,
    }


def eval_video(video, kp_model, line_model, mixer, window_size, device, stride, max_frames):
    files = sorted(Path(FRAMES, f"SNGS-{video}", "img1").glob("*.jpg"))
    if max_frames:
        files = files[:max_frames]
    gt = load_gt_lines_for_video(DATA_ROOT, video)
    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    history = []
    rows = {"baseline": [], "input_mixer": []}
    start = time.perf_counter()
    for idx, image_path in enumerate(files):
        image = tfm(Image.open(image_path).convert("RGB"))
        history.append(image)
        if len(history) > window_size:
            history.pop(0)
        if idx % stride != 0:
            continue
        gid = f"3{video}{image_path.stem}"
        if gid not in gt:
            continue
        x = image.unsqueeze(0).to(device)
        win = left_pad_window(history, window_size).unsqueeze(0).to(device)
        with torch.no_grad():
            line_hm = line_model(x)
            base_hm = kp_model(x)
            mixed = mixer(win)
            mix_hm = kp_model(mixed)
        for name, hm in (("baseline", base_hm), ("input_mixer", mix_hm)):
            keypoints = decode_keypoints(hm, line_hm)
            params = solve_params(keypoints)
            scored = score_frame(params, gt[gid])
            if scored is None:
                rows[name].append({"frame": image_path.stem, "point_acc": None, "line_acc": None, "reproj": []})
            else:
                rows[name].append({
                    "frame": image_path.stem,
                    "point_acc": float(scored[0]),
                    "line_acc": float(scored[1]),
                    "reproj": [float(x) for x in scored[2]],
                })
    return {
        "video": video,
        "seconds": time.perf_counter() - start,
        "baseline": summarize(rows["baseline"]),
        "input_mixer": summarize(rows["input_mixer"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--videos", nargs="+", default=["116", "117", "118", "119", "120", "121", "122", "123"])
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    kp_model, line_model = load_hrnets(device)
    mixer, window_size, ck = load_mixer(args.checkpoint, device)
    out = {
        "checkpoint": args.checkpoint,
        "checkpoint_meta": {k: ck.get(k) for k in ["model", "window_size", "dataset", "split", "videos", "epoch", "steps"]},
        "stride": args.stride,
        "videos": {},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for video in args.videos:
        result = eval_video(video, kp_model, line_model, mixer, window_size, device, args.stride, args.max_frames)
        out["videos"][video] = result
        json.dump(out, open(args.out, "w"), indent=2)
        b, m = result["baseline"], result["input_mixer"]
        print(
            f"[{video}] base point={b['point_acc']} line={b['line_acc']} reproj={b['reproj_mean']} | "
            f"mixer point={m['point_acc']} line={m['line_acc']} reproj={m['reproj_mean']} "
            f"seconds={result['seconds']:.1f}",
            flush=True,
        )
    base_all, mix_all = [], []
    for result in out["videos"].values():
        base_all.append(result["baseline"])
        mix_all.append(result["input_mixer"])
    out["aggregate"] = {
        "baseline": {
            "point_acc": float(np.mean([x["point_acc"] for x in base_all if x["point_acc"] is not None])),
            "line_acc": float(np.mean([x["line_acc"] for x in base_all if x["line_acc"] is not None])),
            "reproj_mean": float(np.mean([x["reproj_mean"] for x in base_all if x["reproj_mean"] is not None])),
        },
        "input_mixer": {
            "point_acc": float(np.mean([x["point_acc"] for x in mix_all if x["point_acc"] is not None])),
            "line_acc": float(np.mean([x["line_acc"] for x in mix_all if x["line_acc"] is not None])),
            "reproj_mean": float(np.mean([x["reproj_mean"] for x in mix_all if x["reproj_mean"] is not None])),
        },
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print("DONE", json.dumps(out["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
