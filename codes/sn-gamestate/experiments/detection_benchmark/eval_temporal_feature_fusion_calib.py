"""Evaluate TemporalHRNetFeatureFusion against frozen NBJW baseline."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from sn_gamestate.temporal_hrnet import TemporalHRNetFeatureFusion
from experiments.detection_benchmark import eval_temporal_input_mixer_calib as base


def load_feature_model(path, kp_model, device):
    ck = torch.load(path, map_location=device)
    model = TemporalHRNetFeatureFusion(
        kp_model,
        level=ck["fusion_level"],
        window_size=int(ck["window_size"]),
        residual_scale=float(ck.get("residual_scale", 1.0)),
        head_lora_rank=int(ck.get("head_lora_rank", 0)),
        head_lora_scale=float(ck.get("head_lora_scale", 0.05)),
        freeze_hrnet=not bool(ck.get("full_finetune", False)),
    )
    if ck.get("full_finetune"):
        model.load_state_dict(ck["state_dict"], strict=False)
    else:
        model.load_adapter_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, int(ck["window_size"]), ck


def eval_video(video, kp_model, line_model, model, window_size, device, stride, max_frames):
    files = sorted(Path(base.FRAMES, f"SNGS-{video}", "img1").glob("*.jpg"))
    if max_frames:
        files = files[:max_frames]
    gt = base.load_gt_lines_for_video(base.DATA_ROOT, video)
    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    history = []
    rows = {"baseline": [], "feature_fusion": []}
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
        win = base.left_pad_window(history, window_size).unsqueeze(0).to(device)
        with torch.no_grad():
            line_hm = line_model(x)
            base_hm = kp_model(x)
            fused_hm = model(win)
        for name, hm in (("baseline", base_hm), ("feature_fusion", fused_hm)):
            keypoints = base.decode_keypoints(hm, line_hm)
            params = base.solve_params(keypoints)
            scored = base.score_frame(params, gt[gid])
            if scored is None:
                rows[name].append({"frame": image_path.stem, "point_acc": None, "line_acc": None, "reproj": [], "params": {}})
            else:
                rows[name].append({
                    "frame": image_path.stem,
                    "point_acc": float(scored[0]),
                    "line_acc": float(scored[1]),
                    "reproj": [float(x) for x in scored[2]],
                    "params": params,
                })
    out = {
        "video": video,
        "seconds": time.perf_counter() - start,
        "baseline": base.summarize(rows["baseline"]),
        "feature_fusion": base.summarize(rows["feature_fusion"]),
    }
    out["baseline"]["smooth_mean"] = smooth_mean(rows["baseline"])
    out["feature_fusion"]["smooth_mean"] = smooth_mean(rows["feature_fusion"])
    return out


def flatten_params(params):
    values = {}

    def walk(prefix, value):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), item)
        elif isinstance(value, (int, float, np.number)) and np.isfinite(float(value)):
            values[prefix] = float(value)
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                walk(f"{prefix}.{idx}", item)

    walk("", params or {})
    return values


def smooth_mean(rows):
    vectors = [flatten_params(row.get("params")) for row in rows if row.get("params")]
    jumps = []
    for prev, cur in zip(vectors, vectors[1:]):
        keys = sorted(set(prev) | set(cur))
        if keys:
            jumps.append(float(np.linalg.norm([cur.get(k, 0.0) - prev.get(k, 0.0) for k in keys])))
    return float(np.mean(jumps)) if jumps else None


def mean_metric(items, key):
    vals = [x[key] for x in items if x[key] is not None]
    return float(np.mean(vals)) if vals else None


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
    baseline_kp_model, line_model = base.load_hrnets(device)
    fusion_kp_model, _ = base.load_hrnets(device)
    model, window_size, ck = load_feature_model(args.checkpoint, fusion_kp_model, device)
    out = {
        "checkpoint": args.checkpoint,
        "checkpoint_meta": {k: ck.get(k) for k in ["model", "fusion_level", "window_size", "split", "videos", "epoch", "steps", "continuity_weight", "residual_scale", "head_lora_rank", "head_lora_scale", "full_finetune", "loss_weights", "hrnet_lr", "adapter_lr"]},
        "stride": args.stride,
        "decode_thresholds": {"kp": base.KP_DECODE_THRESHOLD, "line": base.LINE_DECODE_THRESHOLD},
        "videos": {},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for video in args.videos:
        result = eval_video(video, baseline_kp_model, line_model, model, window_size, device, args.stride, args.max_frames)
        out["videos"][video] = result
        json.dump(out, open(args.out, "w"), indent=2)
        b, m = result["baseline"], result["feature_fusion"]
        print(
            f"[{video}] base point={b['point_acc']} line={b['line_acc']} reproj={b['reproj_mean']} | "
            f"feature point={m['point_acc']} line={m['line_acc']} reproj={m['reproj_mean']} "
            f"seconds={result['seconds']:.1f}",
            flush=True,
        )

    base_all = [result["baseline"] for result in out["videos"].values()]
    fused_all = [result["feature_fusion"] for result in out["videos"].values()]
    out["aggregate"] = {
        "baseline": {
            "point_acc": mean_metric(base_all, "point_acc"),
            "line_acc": mean_metric(base_all, "line_acc"),
            "reproj_mean": mean_metric(base_all, "reproj_mean"),
            "smooth_mean": mean_metric(base_all, "smooth_mean"),
        },
        "feature_fusion": {
            "point_acc": mean_metric(fused_all, "point_acc"),
            "line_acc": mean_metric(fused_all, "line_acc"),
            "reproj_mean": mean_metric(fused_all, "reproj_mean"),
            "smooth_mean": mean_metric(fused_all, "smooth_mean"),
        },
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print("DONE", json.dumps(out["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
