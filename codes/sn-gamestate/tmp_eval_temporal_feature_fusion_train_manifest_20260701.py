"""Evaluate lossfix temporal HRNet checkpoints on SoccerNetGS_2024_nbjw train manifest."""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from experiments.detection_benchmark import eval_temporal_input_mixer_calib as base
from experiments.detection_benchmark.eval_temporal_feature_fusion_calib import load_feature_model


RUNS = [
    ("fullft_stage1_nomotion_k3", "fullft_stage1_nomotion_k3/checkpoints/fullft_stage1_nomotion_k3.pt"),
    ("fullft_last_nomotion_k3", "fullft_last_nomotion_k3/checkpoints/fullft_last_nomotion_k3.pt"),
    ("fullft_last_motion_k3", "fullft_last_motion_k3/checkpoints/fullft_last_motion_k3.pt"),
    ("fullft_stage1_motion_k3", "fullft_stage1_motion_k3/checkpoints/fullft_stage1_motion_k3.pt"),
]


def read_manifest(dataset_root, split, stride, max_videos, max_frames_per_video):
    by_video = defaultdict(list)
    path = Path(dataset_root) / f"{split}_manifest.tsv"
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            by_video[row["video"]].append(row)
    out = {}
    for video in sorted(by_video):
        rows = sorted(by_video[video], key=lambda r: int(r["image_id"]))
        rows = rows[::stride]
        if max_frames_per_video:
            rows = rows[:max_frames_per_video]
        out[video] = rows
        if max_videos and len(out) >= max_videos:
            break
    return out


def smooth_mean(rows):
    vectors = [base_flatten(row.get("params")) for row in rows if row.get("params")]
    jumps = []
    for prev, cur in zip(vectors, vectors[1:]):
        keys = sorted(set(prev) | set(cur))
        if keys:
            jumps.append(float(np.linalg.norm([cur.get(k, 0.0) - prev.get(k, 0.0) for k in keys])))
    return float(np.mean(jumps)) if jumps else None


def base_flatten(params):
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


def summarize(items):
    point = [x["point_acc"] for x in items if x["point_acc"] is not None]
    line = [x["line_acc"] for x in items if x["line_acc"] is not None]
    reproj = [v for x in items for v in x["reproj"]]
    return {
        "point_acc": float(np.mean(point)) if point else None,
        "line_acc": float(np.mean(line)) if line else None,
        "reproj_mean": float(np.mean(reproj)) if reproj else None,
        "reproj_median": float(np.median(reproj)) if reproj else None,
        "smooth_mean": smooth_mean(items),
        "n_scored": len(point),
        "n_total": len(items),
        "completeness": len(point) / len(items) if items else None,
    }


def score_one(kp_hm, line_hm, gt_lines, frame_id):
    keypoints = base.decode_keypoints(kp_hm, line_hm)
    params = base.solve_params(keypoints)
    scored = base.score_frame(params, gt_lines)
    if scored is None:
        return {"frame": frame_id, "point_acc": None, "line_acc": None, "reproj": [], "params": {}}
    return {
        "frame": frame_id,
        "point_acc": float(scored[0]),
        "line_acc": float(scored[1]),
        "reproj": [float(x) for x in scored[2]],
        "params": params,
    }


def eval_run(run_name, ckpt_path, frames_by_video, device, out_path):
    baseline_kp, line_model = base.load_hrnets(device)
    kp_for_model, _line = base.load_hrnets(device)
    model, window_size, ck = load_feature_model(str(ckpt_path), kp_for_model, device)
    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    out = {
        "run": run_name,
        "checkpoint": str(ckpt_path),
        "checkpoint_meta": {k: ck.get(k) for k in ["fusion_level", "window_size", "full_finetune", "loss_weights", "hrnet_lr", "adapter_lr"]},
        "videos": {},
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    start_all = time.perf_counter()
    for video, rows in frames_by_video.items():
        history = []
        scored = {"baseline": [], "feature_fusion": []}
        start = time.perf_counter()
        for row in rows:
            stem = Path(row["dst_image"]).with_suffix("")
            image = tfm(Image.open(str(stem) + ".jpg").convert("RGB"))
            history.append(image)
            if len(history) > window_size:
                history.pop(0)
            x = image.unsqueeze(0).to(device)
            win = base.left_pad_window(history, window_size).unsqueeze(0).to(device)
            gt_lines = json.load(open(str(stem) + ".json"))
            if "Goal left post left" in gt_lines:
                gt_lines["Goal left post left "] = gt_lines.pop("Goal left post left")
            with torch.no_grad():
                line_hm = line_model(x)
                base_hm = baseline_kp(x)
                fused_hm = model(win)
            scored["baseline"].append(score_one(base_hm, line_hm, gt_lines, row["image_id"]))
            scored["feature_fusion"].append(score_one(fused_hm, line_hm, gt_lines, row["image_id"]))
        out["videos"][video] = {
            "seconds": time.perf_counter() - start,
            "baseline": summarize(scored["baseline"]),
            "feature_fusion": summarize(scored["feature_fusion"]),
        }
        json.dump(out, open(out_path, "w"), indent=2)
        b = out["videos"][video]["baseline"]
        m = out["videos"][video]["feature_fusion"]
        print(
            f"[{run_name} {video}] base p={b['point_acc']} l={b['line_acc']} r={b['reproj_mean']} sm={b['smooth_mean']} | "
            f"model p={m['point_acc']} l={m['line_acc']} r={m['reproj_mean']} sm={m['smooth_mean']} frames={b['n_total']}",
            flush=True,
        )
    base_all = [x["baseline"] for x in out["videos"].values()]
    model_all = [x["feature_fusion"] for x in out["videos"].values()]
    out["aggregate"] = {
        "seconds": time.perf_counter() - start_all,
        "baseline": mean_summary(base_all),
        "feature_fusion": mean_summary(model_all),
    }
    json.dump(out, open(out_path, "w"), indent=2)
    return out


def mean_summary(items):
    def m(key):
        vals = [x.get(key) for x in items if x.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    return {
        "point_acc": m("point_acc"),
        "line_acc": m("line_acc"),
        "reproj_mean": m("reproj_mean"),
        "smooth_mean": m("smooth_mean"),
        "n_total": int(sum(x.get("n_total", 0) for x in items)),
        "n_scored": int(sum(x.get("n_scored", 0) for x in items)),
    }


def write_summary(results, out_dir):
    rows = []
    for run, data in results.items():
        b = data["aggregate"]["baseline"]
        m = data["aggregate"]["feature_fusion"]
        rows.append({
            "run": run,
            "base_point": b["point_acc"],
            "model_point": m["point_acc"],
            "base_line": b["line_acc"],
            "model_line": m["line_acc"],
            "base_reproj": b["reproj_mean"],
            "model_reproj": m["reproj_mean"],
            "base_smooth": b["smooth_mean"],
            "model_smooth": m["smooth_mean"],
            "frames": b["n_total"],
        })
    out_dir = Path(out_dir)
    with (out_dir / "aggregate_train_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Train Manifest Evaluation", "", "| run | base point | model point | base line | model line | base reproj | model reproj | base smooth | model smooth | frames |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['run']} | {fmt(r['base_point'])} | {fmt(r['model_point'])} | {fmt(r['base_line'])} | {fmt(r['model_line'])} | "
            f"{fmt(r['base_reproj'],2)} | {fmt(r['model_reproj'],2)} | {fmt(r['base_smooth'],1)} | {fmt(r['model_smooth'],1)} | {r['frames']} |"
        )
    (out_dir / "aggregate_train_manifest.md").write_text("\n".join(lines) + "\n")


def fmt(v, nd=4):
    return "NA" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/remote-home/jiayuanrao/yishan/datasets/SoccerNetGS_2024_nbjw")
    ap.add_argument("--split", default="train")
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--max-frames-per-video", type=int, default=0)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    frames_by_video = read_manifest(args.dataset_root, args.split, args.stride, args.max_videos, args.max_frames_per_video)
    results = {}
    for run, rel_ckpt in RUNS:
        out_path = Path(args.out_dir) / run / "results.json"
        results[run] = eval_run(run, Path(args.ckpt_root) / rel_ckpt, frames_by_video, device, out_path)
    write_summary(results, args.out_dir)
    print("DONE", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
