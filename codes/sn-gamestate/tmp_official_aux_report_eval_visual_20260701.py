import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

from experiments.detection_benchmark import eval_temporal_input_mixer_calib as base
from experiments.detection_benchmark.eval_temporal_feature_fusion_calib import load_feature_model as base_load_feature_model
from nbjw_calib.utils.utils_heatmap import (
    coords_to_dict,
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
)
from nbjw_calib.utils.utils_keypoints import KeypointsDB
from sn_gamestate.temporal_hrnet import TemporalHRNetFeatureFusion


RUNS = [
    ("baseline", None),
    ("fullft_offaux_last_motion_k3", "fullft_offaux_last_motion_k3/checkpoints/fullft_offaux_last_motion_k3.pt"),
    ("fullft_offaux_last_nomotion_k3", "fullft_offaux_last_nomotion_k3/checkpoints/fullft_offaux_last_nomotion_k3.pt"),
    ("fullft_offaux_stage1_motion_k3", "fullft_offaux_stage1_motion_k3/checkpoints/fullft_offaux_stage1_motion_k3.pt"),
    ("fullft_offaux_stage1_nomotion_k3", "fullft_offaux_stage1_nomotion_k3/checkpoints/fullft_offaux_stage1_nomotion_k3.pt"),
    ("fullft_cached_k5_last_motion_residual_balanced_e5", "fullft_cached_k5_last_motion_residual_balanced_e5/latest.pt"),
    ("fullft_cached_k5_stage1_motion_residual_balanced_e5", "fullft_cached_k5_stage1_motion_residual_balanced_e5/latest.pt"),
    ("fullft_cached_k15_stage1_motion_lastpair_fast_e5", "fullft_cached_k15_stage1_motion_lastpair_fast_e5/latest.pt"),
    ("fullft_cached_k15_stage1_motion_lastpair_fast_e5_restart_stepckpt", "fullft_cached_k15_stage1_motion_lastpair_fast_e5_restart_stepckpt/latest.pt"),
    ("fullft_cached_k5_last_motion_lastpair_fast_e5", "fullft_cached_k5_last_motion_lastpair_fast_e5/latest.pt"),
    ("fullft_cached_k5_stage1_motion_lastpair_fast_e5", "fullft_cached_k5_stage1_motion_lastpair_fast_e5/latest.pt"),
]


def fmt(v, nd=4):
    return "NA" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def mean(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def load_feature_model(path, kp_model, device):
    ck = torch.load(path, map_location=device)
    if "fusion_level" in ck:
        return base_load_feature_model(path, kp_model, device)
    cfg = ck.get("config", {})
    model = TemporalHRNetFeatureFusion(
        kp_model,
        level=cfg["fusion_level"],
        window_size=int(cfg["window_size"]),
        residual_scale=float(cfg.get("residual_scale", 1.0)),
        freeze_hrnet=False,
    )
    model.load_state_dict(ck["state_dict"], strict=False)
    model.to(device).eval()
    return model, int(cfg["window_size"]), ck


def camera_smooth_l2(rows):
    def flatten(params):
        out = {}
        def walk(prefix, value):
            if isinstance(value, dict):
                for k, v in value.items():
                    walk(f"{prefix}.{k}" if prefix else str(k), v)
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    walk(f"{prefix}.{i}", v)
            else:
                try:
                    f = float(value)
                    if np.isfinite(f):
                        out[prefix] = f
                except Exception:
                    pass
        walk("", params or {})
        return out

    vecs = [flatten(r.get("params")) for r in rows if r.get("params")]
    jumps = []
    for a, b in zip(vecs, vecs[1:]):
        keys = sorted(set(a) | set(b))
        if keys:
            jumps.append(float(np.linalg.norm([b.get(k, 0.0) - a.get(k, 0.0) for k in keys])))
    if not jumps:
        return None, None
    return float(np.mean(jumps)), float(np.percentile(jumps, 95))


def summarize(rows):
    reproj = [x for r in rows for x in (r.get("reproj") or [])]
    scored = [r for r in rows if r.get("reproj_mean") is not None]
    smooth_mean, smooth_p95 = camera_smooth_l2(rows)
    jac5 = mean([1.0 if x <= 5.0 else 0.0 for x in reproj])
    jac10 = mean([1.0 if x <= 10.0 else 0.0 for x in reproj])
    jac15 = mean([1.0 if x <= 15.0 else 0.0 for x in reproj])
    jac20 = mean([1.0 if x <= 20.0 else 0.0 for x in reproj])
    cr = len(scored) / len(rows) if rows else None
    return {
        "point_acc": mean([r.get("point_acc") for r in rows]),
        "line_acc": mean([r.get("line_acc") for r in rows]),
        "reproj_mean": mean([r.get("reproj_mean") for r in rows]),
        "smooth_mean": smooth_mean,
        "JaC@5": jac5,
        "JaC@10": jac10,
        "JaC@15": jac15,
        "JaC@20": jac20,
        "MRE": mean(reproj),
        "CR": cr,
        "Final Score": (cr * jac5) if cr is not None and jac5 is not None else None,
        "camera_smooth_l2_mean": smooth_mean,
        "camera_smooth_l2_p95": smooth_p95,
        "n_total": len(rows),
        "n_scored": len(scored),
    }


def get_split_paths(split):
    root = Path("/remote-home/jiayuanrao/yishan/sn-gamestate/data/SoccerNetGS") / split
    return root, root


def list_videos(split, videos):
    frames_root, _ = get_split_paths(split)
    if videos:
        return [str(v).replace("SNGS-", "") for v in videos]
    return sorted(p.name.replace("SNGS-", "") for p in frames_root.glob("SNGS-*") if (p / "img1").exists())


def image_id_map(data_root, video):
    labels = Path(data_root) / f"SNGS-{video}" / "Labels-GameState.json"
    if not labels.exists():
        return {}
    try:
        data = json.load(open(labels))
        return {Path(item["file_name"]).stem: str(item["image_id"]) for item in data.get("images", [])}
    except Exception:
        return {}


def score_hm(kp_hm, line_hm, gt_lines):
    keypoints = base.decode_keypoints(kp_hm, line_hm)
    params = base.solve_params(keypoints)
    scored = base.score_frame(params, gt_lines)
    if scored is None:
        return {"point_acc": None, "line_acc": None, "reproj": [], "reproj_mean": None, "params": params}
    reproj = [float(x) for x in scored[2]]
    return {
        "point_acc": float(scored[0]),
        "line_acc": float(scored[1]),
        "reproj": reproj,
        "reproj_mean": float(np.mean(reproj)) if reproj else None,
        "params": params,
    }


def wrap180(deg):
    return ((float(deg) + 180.0) % 360.0) - 180.0


def signed_angle_to_midline(params):
    pan = params.get("pan_degrees") if isinstance(params, dict) else None
    if pan is None:
        return None
    # Middle-line axis in the NBJW pan frame is represented by +/-90 deg.
    # Keep the sign of the nearest-axis deviation so left/right views are not folded together.
    candidates = [wrap180(float(pan) - 90.0), wrap180(float(pan) + 90.0)]
    return float(min(candidates, key=lambda x: abs(x)))


def folded_angle_to_midline(params):
    signed = signed_angle_to_midline(params)
    return None if signed is None else abs(float(signed))


def eval_one_split(args):
    device = torch.device(args.device)
    frames_root, data_root = get_split_paths(args.split)
    videos = list_videos(args.split, args.videos)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_kp, line_model = base.load_hrnets(device)
    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    all_results = {}
    frame_rows = []

    selected = [r for r in RUNS if not args.runs or r[0] in set(args.runs) or (r[0] == "baseline" and "baseline" in set(args.runs))]
    if not any(r[0] == "baseline" for r in selected):
        selected = [("baseline", None)] + selected

    loaded = {"baseline": (None, 1)}
    for run_name, rel in selected:
        if run_name == "baseline":
            continue
        kp_for_model, _ = base.load_hrnets(device)
        model, window_size, ck = load_feature_model(str(Path(args.ckpt_root) / rel), kp_for_model, device)
        loaded[run_name] = (model, window_size)

    for video in videos:
        files = sorted((frames_root / f"SNGS-{video}" / "img1").glob("*.jpg"))
        if args.max_frames:
            files = files[:args.max_frames]
        gt = base.load_gt_lines_for_video(str(data_root), video)
        id_map = image_id_map(data_root, video)
        histories = {name: [] for name, _ in selected}
        scored_by_run = {name: [] for name, _ in selected}
        start = time.perf_counter()
        for idx, image_path in enumerate(files):
            pil = Image.open(image_path).convert("RGB")
            image = tfm(pil)
            for name in histories:
                histories[name].append(image)
                _, ws = loaded[name]
                if len(histories[name]) > ws:
                    histories[name].pop(0)
            if idx % args.stride != 0:
                continue
            gid = id_map.get(image_path.stem, f"3{video}{image_path.stem}")
            if gid not in gt:
                continue
            gt_lines = gt[gid]
            x = image.unsqueeze(0).to(device)
            with torch.no_grad():
                line_hm = line_model(x)
                base_hm = baseline_kp(x)
            for name, _rel in selected:
                if name == "baseline":
                    hm = base_hm
                    ws = 1
                else:
                    model, ws = loaded[name]
                    win = base.left_pad_window(histories[name], ws).unsqueeze(0).to(device)
                    with torch.no_grad():
                        hm = model(win)
                s = score_hm(hm, line_hm, gt_lines)
                row = {
                    "run": name,
                    "split": args.split,
                    "video": video,
                    "frame": image_path.stem,
                    "image_path": str(image_path),
                    "point_acc": s["point_acc"],
                    "line_acc": s["line_acc"],
                    "reproj_mean": s["reproj_mean"],
                    "reproj": s["reproj"],
                    "angle_to_midline_deg": signed_angle_to_midline(s["params"]),
                    "signed_angle_to_midline_deg": signed_angle_to_midline(s["params"]),
                    "folded_angle_to_midline_deg": folded_angle_to_midline(s["params"]),
                    "params": s["params"],
                }
                scored_by_run[name].append(row)
                flat = {k: v for k, v in row.items() if k not in ("params", "reproj")}
                frame_rows.append(flat)
        for name, rows in scored_by_run.items():
            all_results.setdefault(name, {"videos": {}})
            all_results[name]["videos"][video] = summarize(rows)
            print(args.split, video, name, all_results[name]["videos"][video], f"seconds={time.perf_counter()-start:.1f}", flush=True)

    for name in all_results:
        vids = all_results[name]["videos"]
        all_results[name]["aggregate"] = {
            "point_acc": mean([v["point_acc"] for v in vids.values()]),
            "line_acc": mean([v["line_acc"] for v in vids.values()]),
            "reproj_mean": mean([v["reproj_mean"] for v in vids.values()]),
            "smooth_mean": mean([v["smooth_mean"] for v in vids.values()]),
            "JaC@5": mean([v["JaC@5"] for v in vids.values()]),
            "JaC@10": mean([v["JaC@10"] for v in vids.values()]),
            "JaC@15": mean([v["JaC@15"] for v in vids.values()]),
            "JaC@20": mean([v["JaC@20"] for v in vids.values()]),
            "MRE": mean([v["MRE"] for v in vids.values()]),
            "CR": mean([v["CR"] for v in vids.values()]),
            "Final Score": mean([v["Final Score"] for v in vids.values()]),
            "camera_smooth_l2_mean": mean([v["camera_smooth_l2_mean"] for v in vids.values()]),
            "camera_smooth_l2_p95": mean([v["camera_smooth_l2_p95"] for v in vids.values()]),
            "n_total": int(sum(v["n_total"] for v in vids.values())),
            "n_scored": int(sum(v["n_scored"] for v in vids.values())),
        }

    (out_dir / f"{args.split}_results.json").write_text(json.dumps(all_results, indent=2))
    with (out_dir / f"{args.split}_frame_scores.csv").open("w", newline="") as f:
        fields = [
            "run", "split", "video", "frame", "image_path", "point_acc", "line_acc",
            "reproj_mean", "angle_to_midline_deg", "signed_angle_to_midline_deg",
            "folded_angle_to_midline_deg",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(frame_rows)
    write_tables(all_results, out_dir, args.split)
    return all_results


def write_tables(results, out_dir, split):
    rows = []
    names = list(results)
    videos = sorted({v for r in results.values() for v in r["videos"]})
    for video in videos:
        for name in names:
            m = results[name]["videos"][video]
            rows.append({"run": name, "video": video, **m})
    with (out_dir / f"{split}_per_video_metrics.csv").open("w", newline="") as f:
        fields = [
            "run", "video", "point_acc", "line_acc", "reproj_mean", "smooth_mean",
            "JaC@5", "JaC@10", "JaC@15", "JaC@20", "MRE", "CR", "Final Score",
            "camera_smooth_l2_mean", "camera_smooth_l2_p95", "n_total", "n_scored",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    lines = [f"# {split} Per-Video Metrics", "", "| run | video | JaC@5 | JaC@10 | JaC@15 | JaC@20 | MRE | CR | Final Score | camera smooth mean | camera smooth p95 | frames |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['video']} | {fmt(r['JaC@5'])} | {fmt(r['JaC@10'])} | {fmt(r['JaC@15'])} | {fmt(r['JaC@20'])} | {fmt(r['MRE'],2)} | {fmt(r['CR'])} | {fmt(r['Final Score'])} | {fmt(r['camera_smooth_l2_mean'],1)} | {fmt(r['camera_smooth_l2_p95'],1)} | {r['n_total']} |")
    lines += ["", f"# {split} Aggregate", "", "| run | JaC@5 | JaC@10 | JaC@15 | JaC@20 | MRE | CR | Final Score | camera smooth mean | camera smooth p95 | frames |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        a = results[name]["aggregate"]
        lines.append(f"| {name} | {fmt(a['JaC@5'])} | {fmt(a['JaC@10'])} | {fmt(a['JaC@15'])} | {fmt(a['JaC@20'])} | {fmt(a['MRE'],2)} | {fmt(a['CR'])} | {fmt(a['Final Score'])} | {fmt(a['camera_smooth_l2_mean'],1)} | {fmt(a['camera_smooth_l2_p95'],1)} | {a['n_total']} |")
    (out_dir / f"{split}_metrics.md").write_text("\n".join(lines) + "\n")


def norm_heatmap(hm):
    arr = hm.detach().float().cpu().numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr, [1, 99.5])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)


def heat_overlay(image, heat, color, alpha=0.42):
    heat_img = Image.fromarray((norm_heatmap(heat) * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    rgba = Image.new("RGBA", image.size, color + (0,))
    rgba.putalpha(heat_img.point(lambda v: int(v * alpha)))
    return Image.alpha_composite(image.convert("RGBA"), rgba).convert("RGB")


def label(image, lines):
    draw = ImageDraw.Draw(image)
    h = 18 + 14 * len(lines)
    draw.rectangle((0, 0, image.size[0], h), fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((8, 5 + 14 * i), line, fill=(255, 255, 255))


def draw_cross(draw, x, y, color, r=5, width=2):
    draw.line((x - r, y, x + r, y), fill=color, width=width)
    draw.line((x, y - r, x, y + r), fill=color, width=width)


def draw_circle(draw, x, y, color, r=5, width=2):
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=width)


def gt_heatmap_from_lines(lines, image_tensor):
    try:
        gt, mask = KeypointsDB(lines or {}, image_tensor.cpu()).get_tensor_w_mask()
        return torch.from_numpy(gt).unsqueeze(0).float()
    except Exception:
        return None


def gt_heatmap_from_nbjw_json(video, frame, image_tensor):
    json_path = Path("/remote-home/jiayuanrao/yishan/datasets/SoccerNetGS_2024_nbjw/test") / f"3{video}{frame}.json"
    if not json_path.exists():
        return None
    try:
        data = json.load(open(json_path))
        if "Goal left post left" in data:
            data["Goal left post left "] = data.pop("Goal left post left")
        gt, _mask = KeypointsDB(data or {}, image_tensor.cpu()).get_tensor_w_mask()
        return torch.from_numpy(gt).unsqueeze(0).float()
    except Exception:
        return None


def draw_gt_lines(image, lines, color=(0, 255, 0), width=2):
    draw = ImageDraw.Draw(image)
    for pts in (lines or {}).values():
        xy = []
        for p in pts:
            try:
                x, y = float(p["x"]), float(p["y"])
                if x <= 1.5 and y <= 1.5:
                    x, y = x * 960, y * 540
                else:
                    x, y = x / 2.0, y / 2.0
                xy.append((x, y))
            except Exception:
                pass
        if len(xy) >= 2:
            draw.line(xy, fill=color, width=width)


def draw_gt_points(image, gt_hm, threshold=0.1):
    if gt_hm is None:
        return 0
    coords = get_keypoints_from_heatmap_batch_maxpool(gt_hm[:, :-1])
    kp = coords_to_dict(coords, threshold=threshold)[0]
    draw = ImageDraw.Draw(image)
    for ch, item in sorted(kp.items()):
        x, y = float(item["x"]), float(item["y"])
        draw_circle(draw, x, y, (0, 255, 0), r=6, width=2)
        draw.text((x + 7, y + 3), f"G{ch}", fill=(0, 255, 0))
    return len(kp)


def draw_kp_peaks(image, hm, color, threshold=0.1449):
    coords = get_keypoints_from_heatmap_batch_maxpool(hm[:, :-1])
    kp = coords_to_dict(coords, threshold=threshold)[0]
    draw = ImageDraw.Draw(image)
    for ch, item in sorted(kp.items()):
        x, y = float(item["x"]), float(item["y"])
        draw_cross(draw, x, y, color)
        draw.text((x + 6, y - 8), str(ch), fill=color)
    return len(kp)


def draw_line_endpoints(image, line_hm, threshold=0.1):
    coords = get_keypoints_from_heatmap_batch_maxpool_l(line_hm[:, :-1])
    lines = coords_to_dict(coords, threshold=threshold)[0]
    draw = ImageDraw.Draw(image)
    for ch, item in sorted(lines.items()):
        try:
            x1, y1 = float(item["x_1"]), float(item["y_1"])
            x2, y2 = float(item["x_2"]), float(item["y_2"])
        except Exception:
            continue
        draw.line((x1, y1, x2, y2), fill=(255, 230, 0), width=2)
        draw_cross(draw, x1, y1, (255, 230, 0))
        draw_cross(draw, x2, y2, (255, 230, 0))
        draw.text((x1 + 6, y1 - 8), str(ch), fill=(255, 230, 0))
    return len(lines)


def render_visuals(args):
    out_dir = Path(args.out_dir)
    csv_path = out_dir / "test_frame_scores.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    by_run_video = defaultdict(list)
    with csv_path.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["reproj_mean"] in ("", "None", "nan"):
                continue
            r["reproj_mean"] = float(r["reproj_mean"])
            by_run_video[(r["run"], r["video"])].append(r)

    selections = []
    for key, rows in by_run_video.items():
        rows = sorted(rows, key=lambda r: int(r["frame"]))
        best = min(rows, key=lambda r: r["reproj_mean"])
        worst = max(rows, key=lambda r: r["reproj_mean"])
        for tag, center in (("best", best), ("worst", worst)):
            c = int(center["frame"])
            for frame in [max(1, c - 2), max(1, c - 1), c]:
                selections.append({"run": key[0], "video": key[1], "tag": tag, "center": f"{c:06d}", "frame": f"{frame:06d}"})

    device = torch.device(args.device)
    baseline_kp, line_model = base.load_hrnets(device)
    models = {"baseline": (None, 1)}
    needed_runs = sorted({key[0] for key in by_run_video if key[0] != "baseline"})
    if args.runs:
        allowed = set(args.runs)
        needed_runs = [r for r in needed_runs if r in allowed]
    for name, rel in RUNS[1:]:
        if name not in needed_runs:
            continue
        kp_for_model, _ = base.load_hrnets(device)
        models[name] = load_feature_model(str(Path(args.ckpt_root) / rel), kp_for_model, device)[:2]
    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    gt_cache = {}
    summary = []
    vis_dir = out_dir / "visual_best_worst_k3"
    for sel in selections:
        video, frame, run = sel["video"], sel["frame"], sel["run"]
        image_path = Path("/remote-home/jiayuanrao/yishan/sn-gamestate/data/SoccerNetGS/test") / f"SNGS-{video}" / "img1" / f"{frame}.jpg"
        pil = Image.open(image_path).convert("RGB")
        image = pil.resize((960, 540))
        image_tensor = tfm(pil)
        x = image_tensor.unsqueeze(0).to(device)
        if video not in gt_cache:
            gt_cache[video] = base.load_gt_lines_for_video("/remote-home/jiayuanrao/yishan/sn-gamestate/data/SoccerNetGS/test", video)
        gt_lines = gt_cache[video].get(f"3{video}{frame}", {})
        with torch.no_grad():
            line_hm = line_model(x)
            base_hm = baseline_kp(x)
        if run == "baseline":
            model_hm = base_hm
        else:
            model, ws = models[run]
            files = sorted(image_path.parent.glob("*.jpg"))
            idx = files.index(image_path)
            frames = [tfm(Image.open(p).convert("RGB")) for p in files[max(0, idx - ws + 1):idx + 1]]
            win = base.left_pad_window(frames, ws).unsqueeze(0).to(device)
            with torch.no_grad():
                model_hm = model(win)
        gt_hm = gt_heatmap_from_nbjw_json(video, frame, image_tensor)
        base_metrics = score_hm(base_hm, line_hm, gt_lines)
        model_metrics = score_hm(model_hm, line_hm, gt_lines)

        p1 = heat_overlay(image, base_hm[0, :-1].max(0).values, (255, 0, 0))
        draw_gt_lines(p1, gt_lines, width=1)
        n_gt = draw_gt_points(p1, gt_hm)
        n_base = draw_kp_peaks(p1, base_hm, (0, 255, 255))
        label(p1, [f"baseline {video}/{frame}: cyan x pred, green o GT", f"p={fmt(base_metrics['point_acc'])} l={fmt(base_metrics['line_acc'])} r={fmt(base_metrics['reproj_mean'],2)}", f"pred_kp={n_base} gt_kp={n_gt}"])

        p2 = heat_overlay(image, model_hm[0, :-1].max(0).values, (255, 0, 0))
        draw_gt_lines(p2, gt_lines, width=1)
        n_gt2 = draw_gt_points(p2, gt_hm)
        n_model = draw_kp_peaks(p2, model_hm, (255, 230, 0))
        label(p2, [f"{run} {sel['tag']} center={sel['center']}: yellow x pred, green o GT", f"p={fmt(model_metrics['point_acc'])} l={fmt(model_metrics['line_acc'])} r={fmt(model_metrics['reproj_mean'],2)}", f"pred_kp={n_model} gt_kp={n_gt2}"])

        p3 = heat_overlay(image, line_hm[0, :-1].max(0).values, (0, 128, 255))
        draw_gt_lines(p3, gt_lines, width=1)
        n_line = draw_line_endpoints(p3, line_hm, threshold=0.1)
        label(p3, ["frozen line HRNet heatmap/endpoints @0.1", f"pred_lines={n_line}; GT lines green"])

        p4 = image.copy()
        draw_gt_lines(p4, gt_lines, width=3)
        draw_gt_points(p4, gt_hm)
        label(p4, ["GT field lines + supervised GT keypoints", f"gt_lines={len(gt_lines or {})} gt_kp={n_gt}"])

        canvas = Image.new("RGB", (1920, 1080), (20, 20, 20))
        canvas.paste(p1, (0, 0)); canvas.paste(p2, (960, 0)); canvas.paste(p3, (0, 540)); canvas.paste(p4, (960, 540))
        out_path = vis_dir / run / f"SNGS-{video}_{sel['tag']}_center{sel['center']}_frame{frame}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, quality=92)
        summary.append({**sel, "file": str(out_path), "base_reproj": base_metrics["reproj_mean"], "model_reproj": model_metrics["reproj_mean"]})
        print("VIS", out_path, flush=True)

    with (vis_dir / "visual_summary.csv").open("w", newline="") as f:
        fields = ["run", "video", "tag", "center", "frame", "base_reproj", "model_reproj", "file"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(summary)
    (vis_dir / "README.md").write_text(
        "# Best/Worst K=3 Visualizations\n\n"
        "Top-left: baseline point heatmap on RGB, cyan x predicted point peaks, green o GT points.\n"
        "Top-right: selected model point heatmap on RGB, yellow x predicted point peaks, green o GT points.\n"
        "Bottom-left: frozen line HRNet heatmap and decoded endpoints at threshold 0.1, GT lines in green.\n"
        "Bottom-right: GT field lines and GT keypoints.\n"
        "For each run/video, best and worst centers are selected by that run's sampled per-frame reproj_mean; each center exports the K=3 frames ending at that center.\n"
    )


def plot_scatter(args):
    out_dir = Path(args.out_dir)
    rows = []
    with (out_dir / "test_frame_scores.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            try:
                angle = float(r.get("signed_angle_to_midline_deg") or r["angle_to_midline_deg"])
                reproj = float(r["reproj_mean"])
            except Exception:
                continue
            if np.isfinite(angle) and np.isfinite(reproj) and reproj < args.scatter_max_reproj:
                rows.append((r["run"], r["video"], angle, reproj))
    plot_dir = out_dir / "angle_reproj_scatter"
    plot_dir.mkdir(parents=True, exist_ok=True)
    runs = [r[0] for r in rows]
    unique = [name for name, _ in RUNS if name in runs]
    def setup_axes():
        plt.axvline(0, color="black", linewidth=1.0, alpha=0.65)
        plt.xlim(-90, 90)
        plt.ylim(0, args.scatter_max_reproj)
        plt.xlabel("signed optical-axis deviation from pitch middle-line axis (deg, pan proxy)")
        plt.ylabel("per-frame reproj_mean")
        plt.grid(True, alpha=0.25)

    def points(name):
        return [r for r in rows if r[0] == name]

    plt.figure(figsize=(10, 6))
    colors = {
        "baseline": "#1f77b4",
        "fullft_offaux_last_motion_k3": "#d62728",
        "fullft_offaux_last_nomotion_k3": "#ff7f0e",
        "fullft_offaux_stage1_motion_k3": "#2ca02c",
        "fullft_offaux_stage1_nomotion_k3": "#9467bd",
    }
    for name in unique:
        pts = points(name)
        xs = [r[2] for r in pts]
        ys = [r[3] for r in pts]
        plt.scatter(xs, ys, s=18, alpha=0.9, label=name, color=colors.get(name))
    setup_axes()
    plt.ylabel("per-frame reproj_mean")
    plt.title(f"Angle vs Reprojection, SNGS-116..123 stride={args.stride}, reproj<{args.scatter_max_reproj:g}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "signed_angle_reproj_scatter_all_runs.png", dpi=180)
    plt.close()

    base_pts = points("baseline")
    if base_pts:
        xs = [r[2] for r in base_pts]
        ys = [r[3] for r in base_pts]
        plt.figure(figsize=(8, 5))
        plt.scatter(xs, ys, s=20, alpha=0.95, color=colors["baseline"], label="baseline")
        setup_axes()
        plt.title(f"baseline, stride={args.stride}, reproj<{args.scatter_max_reproj:g}")
        plt.tight_layout()
        plt.savefig(plot_dir / "baseline_signed_angle_reproj_scatter.png", dpi=180)
        plt.close()
        plt.figure(figsize=(8, 5))
        plt.hist2d(xs, ys, bins=[36, 25], range=[[-90, 90], [0, args.scatter_max_reproj]], cmap="Blues")
        plt.colorbar(label="frame count")
        setup_axes()
        plt.title("baseline frequency heatmap")
        plt.tight_layout()
        plt.savefig(plot_dir / "baseline_signed_angle_reproj_frequency_heatmap.png", dpi=180)
        plt.close()

    for name in unique:
        if name == "baseline":
            continue
        pts = points(name)
        xs = [r[2] for r in pts]
        ys = [r[3] for r in pts]
        plt.figure(figsize=(8, 5))
        if base_pts:
            plt.scatter([r[2] for r in base_pts], [r[3] for r in base_pts], s=18, alpha=0.85, color="#2b6cb0", label="baseline")
        plt.scatter(xs, ys, s=22, alpha=0.95, color=colors.get(name, "#d62728"), label=name)
        setup_axes()
        plt.title(f"{name} vs baseline, stride={args.stride}, reproj<{args.scatter_max_reproj:g}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{name}_signed_angle_reproj_scatter_vs_baseline.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.hist2d(xs, ys, bins=[36, 25], range=[[-90, 90], [0, args.scatter_max_reproj]], cmap="Oranges")
        plt.colorbar(label="frame count")
        setup_axes()
        plt.title(f"{name} frequency heatmap")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{name}_signed_angle_reproj_frequency_heatmap.png", dpi=180)
        plt.close()
    with (plot_dir / "angle_reproj_points.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "video", "signed_angle_to_midline_deg", "reproj_mean"])
        w.writerows(rows)
    (plot_dir / "README.md").write_text(
        f"Scatter keeps only frames with reproj_mean < {args.scatter_max_reproj:g}. "
        "Angle source: NBJW solved camera params for each evaluated frame. "
        "signed_angle_to_midline_deg is the signed deviation from the nearest middle-line axis (+/-90 deg in the NBJW pan frame), "
        "so 0 deg is the middle-line direction and negative/positive values keep the two sides separate. "
        "This is a reproducible proxy because explicit GT camera pose was not found in the loaded Labels-GameState annotations.\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "visual", "scatter", "all"], default="all")
    ap.add_argument("--split", default="test")
    ap.add_argument("--videos", nargs="*", default=[])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--scatter-max-reproj", type=float, default=25.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.mode in ("eval", "all"):
        eval_one_split(args)
    if args.mode in ("visual", "all") and args.split == "test":
        render_visuals(args)
    if args.mode in ("scatter", "all") and args.split == "test":
        plot_scatter(args)


if __name__ == "__main__":
    main()
