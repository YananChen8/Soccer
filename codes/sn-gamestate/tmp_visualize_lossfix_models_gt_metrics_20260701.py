import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

from experiments.detection_benchmark import eval_temporal_input_mixer_calib as base
from experiments.detection_benchmark.eval_temporal_feature_fusion_calib import load_feature_model
from nbjw_calib.utils.utils_heatmap import (
    coords_to_dict,
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
)
from nbjw_calib.utils.utils_keypoints import KeypointsDB


RUNS = [
    ("fullft_stage1_nomotion_k3", "fullft_stage1_nomotion_k3/checkpoints/fullft_stage1_nomotion_k3.pt"),
    ("fullft_last_nomotion_k3", "fullft_last_nomotion_k3/checkpoints/fullft_last_nomotion_k3.pt"),
    ("fullft_last_motion_k3", "fullft_last_motion_k3/checkpoints/fullft_last_motion_k3.pt"),
    ("fullft_stage1_motion_k3", "fullft_stage1_motion_k3/checkpoints/fullft_stage1_motion_k3.pt"),
]


def norm_heatmap(hm):
    arr = hm.detach().float().cpu().numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr, [1, 99.5])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)


def heat_overlay(image, heat, color):
    heat_img = Image.fromarray((norm_heatmap(heat) * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    rgba = Image.new("RGBA", image.size, color + (0,))
    rgba.putalpha(heat_img.point(lambda v: int(v * 0.65)))
    return Image.alpha_composite(image.convert("RGBA"), rgba).convert("RGB")


def label(image, lines):
    draw = ImageDraw.Draw(image)
    h = 18 + 14 * len(lines)
    draw.rectangle((0, 0, image.size[0], h), fill=(0, 0, 0))
    y = 5
    for line in lines:
        draw.text((8, y), line, fill=(255, 255, 255))
        y += 14


def draw_cross(draw, x, y, color, r=5, width=2):
    draw.line((x - r, y, x + r, y), fill=color, width=width)
    draw.line((x, y - r, x, y + r), fill=color, width=width)


def draw_circle(draw, x, y, color, r=5, width=2):
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=width)


def draw_gt_lines(image, lines, color=(0, 255, 0), width=2):
    draw = ImageDraw.Draw(image)
    for _name, pts in (lines or {}).items():
        xy = []
        for p in pts:
            try:
                xy.append((float(p["x"]) / 2.0, float(p["y"]) / 2.0))
            except Exception:
                pass
        if len(xy) >= 2:
            draw.line(xy, fill=color, width=width)


def gt_heatmap_from_lines(lines, image_tensor):
    try:
        gt, mask = KeypointsDB(lines or {}, image_tensor.cpu()).get_tensor_w_mask()
        return torch.from_numpy(gt).unsqueeze(0).float(), mask
    except Exception:
        return None, None


def draw_gt_points(image, gt_hm, color=(0, 255, 0), threshold=0.1):
    if gt_hm is None:
        return 0
    coords = get_keypoints_from_heatmap_batch_maxpool(gt_hm[:, :-1])
    kp = coords_to_dict(coords, threshold=threshold)[0]
    draw = ImageDraw.Draw(image)
    for channel, item in sorted(kp.items()):
        x, y = float(item["x"]), float(item["y"])
        draw_circle(draw, x, y, color, r=6, width=2)
        draw.text((x + 7, y + 3), f"G{channel}", fill=color)
    return len(kp)


def draw_kp_peaks(image, hm, color=(0, 255, 255), threshold=0.1449):
    coords = get_keypoints_from_heatmap_batch_maxpool(hm[:, :-1])
    kp = coords_to_dict(coords, threshold=threshold)[0]
    draw = ImageDraw.Draw(image)
    for channel, item in sorted(kp.items()):
        x, y = float(item["x"]), float(item["y"])
        draw_cross(draw, x, y, color)
        draw.text((x + 6, y - 8), str(channel), fill=color)
    return len(kp)


def draw_line_endpoints(image, line_hm, threshold=0.1):
    coords = get_keypoints_from_heatmap_batch_maxpool_l(line_hm[:, :-1])
    lines = coords_to_dict(coords, threshold=threshold)[0]
    draw = ImageDraw.Draw(image)
    for channel, item in sorted(lines.items()):
        try:
            x1, y1 = float(item["x_1"]), float(item["y_1"])
            x2, y2 = float(item["x_2"]), float(item["y_2"])
        except Exception:
            continue
        draw.line((x1, y1, x2, y2), fill=(255, 230, 0), width=2)
        draw_cross(draw, x1, y1, (255, 230, 0))
        draw_cross(draw, x2, y2, (255, 230, 0))
        draw.text((x1 + 6, y1 - 8), str(channel), fill=(255, 230, 0))
    return len(lines)


def build_window(image_path, window_size, tfm):
    files = sorted(image_path.parent.glob("*.jpg"))
    idx = files.index(image_path)
    start = max(0, idx - window_size + 1)
    frames = [tfm(Image.open(p).convert("RGB")) for p in files[start : idx + 1]]
    return base.left_pad_window(frames, window_size)


def score_hm(kp_hm, line_hm, gt_lines):
    keypoints = base.decode_keypoints(kp_hm, line_hm)
    params = base.solve_params(keypoints)
    scored = base.score_frame(params, gt_lines)
    if scored is None:
        return {"point": None, "line": None, "reproj": None}
    reproj = [float(x) for x in scored[2]]
    return {
        "point": float(scored[0]),
        "line": float(scored[1]),
        "reproj": float(np.mean(reproj)) if reproj else None,
    }


def fmt_metrics(prefix, metrics, smooth):
    def f(v, nd=3):
        return "NA" if v is None else f"{v:.{nd}f}"
    return (
        f"{prefix} p={f(metrics.get('point'))} l={f(metrics.get('line'))} "
        f"r={f(metrics.get('reproj'), 2)} sm_vid={f(smooth, 1)}"
    )


def load_video_smooth(results_root, run_name, video):
    p = Path(results_root) / run_name / "eval_test116_123_stride20" / "results.json"
    if not p.exists():
        return None, None
    data = json.loads(p.read_text())
    item = data.get("videos", {}).get(str(video), {})
    base_sm = (item.get("baseline") or {}).get("smooth_mean")
    model_sm = (item.get("feature_fusion") or {}).get("smooth_mean")
    return base_sm, model_sm


def render_one(run_name, model, window_size, baseline_kp, line_model, image_path, gt_lines, device, out_path, results_root):
    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    pil = Image.open(image_path).convert("RGB")
    image = pil.resize((960, 540))
    image_tensor = tfm(pil)
    x = image_tensor.unsqueeze(0).to(device)
    win = build_window(image_path, window_size, tfm).unsqueeze(0).to(device)
    with torch.no_grad():
        line_hm = line_model(x)
        base_hm = baseline_kp(x)
        fused_hm = model(win)

    gt_hm, _mask = gt_heatmap_from_lines(gt_lines, image_tensor)
    if gt_hm is not None:
        gt_hm = gt_hm.to(device)
    base_metrics = score_hm(base_hm, line_hm, gt_lines)
    fused_metrics = score_hm(fused_hm, line_hm, gt_lines)
    video = image_path.parent.parent.name.replace("SNGS-", "")
    base_sm, model_sm = load_video_smooth(results_root, run_name, video)

    base_panel = heat_overlay(image, base_hm[0, :-1].max(0).values, (255, 0, 0))
    draw_gt_lines(base_panel, gt_lines, width=1)
    n_gt = draw_gt_points(base_panel, gt_hm, color=(0, 255, 0))
    n_base = draw_kp_peaks(base_panel, base_hm, color=(0, 255, 255))
    label(base_panel, [
        "baseline: cyan x=pred, green o=GT",
        fmt_metrics("base", base_metrics, base_sm),
        f"pred_kp={n_base} gt_kp={n_gt}",
    ])

    fused_panel = heat_overlay(image, fused_hm[0, :-1].max(0).values, (255, 0, 0))
    draw_gt_lines(fused_panel, gt_lines, width=1)
    n_gt2 = draw_gt_points(fused_panel, gt_hm, color=(0, 255, 0))
    n_fused = draw_kp_peaks(fused_panel, fused_hm, color=(255, 230, 0))
    label(fused_panel, [
        f"{run_name}: yellow x=pred, green o=GT",
        fmt_metrics("model", fused_metrics, model_sm),
        f"pred_kp={n_fused} gt_kp={n_gt2}",
    ])

    gt_panel = image.copy()
    draw_gt_lines(gt_panel, gt_lines, width=3)
    draw_gt_points(gt_panel, gt_hm, color=(0, 255, 0))
    label(gt_panel, [f"official GT field lines + GT keypoints", f"gt_lines={len(gt_lines or {})} gt_kp={n_gt}"])

    line_panel = heat_overlay(image, line_hm[0, :-1].max(0).values, (0, 128, 255))
    draw_gt_lines(line_panel, gt_lines, width=1)
    n_line = draw_line_endpoints(line_panel, line_hm, threshold=0.1)
    label(line_panel, ["frozen line HRNet endpoints @0.1", f"pred_lines={n_line}; GT lines green"])

    canvas = Image.new("RGB", (1920, 1080), (20, 20, 20))
    canvas.paste(base_panel, (0, 0))
    canvas.paste(fused_panel, (960, 0))
    canvas.paste(gt_panel, (0, 540))
    canvas.paste(line_panel, (960, 540))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    return {
        "run": run_name,
        "video": video,
        "frame": image_path.stem,
        "base_kp": n_base,
        "model_kp": n_fused,
        "gt_kp": n_gt,
        "line_pred": n_line,
        "gt_lines": len(gt_lines or {}),
        "base_point": base_metrics["point"],
        "model_point": fused_metrics["point"],
        "base_line": base_metrics["line"],
        "model_line": fused_metrics["line"],
        "base_reproj": base_metrics["reproj"],
        "model_reproj": fused_metrics["reproj"],
        "file": str(out_path),
    }


def read_unique_frames(path, limit):
    seen = set()
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (str(row["video"]), str(row["frame"]).zfill(6))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"video": key[0], "frame": key[1]})
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-csv", required=True)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    baseline_kp, line_model = base.load_hrnets(device)
    rows = read_unique_frames(args.frame_csv, args.limit)
    out_dir = Path(args.out_dir)
    all_summary = []

    gt_cache = {}
    for run_name, rel_ckpt in RUNS:
        print("RUN", run_name, flush=True)
        kp_for_model, _line = base.load_hrnets(device)
        model, window_size, _ck = load_feature_model(str(Path(args.ckpt_root) / rel_ckpt), kp_for_model, device)
        for idx, row in enumerate(rows, 1):
            video = str(row["video"])
            frame = str(row["frame"]).zfill(6)
            image_path = Path(base.FRAMES) / f"SNGS-{video}" / "img1" / f"{frame}.jpg"
            if video not in gt_cache:
                gt_cache[video] = base.load_gt_lines_for_video(base.DATA_ROOT, video)
            gt_lines = gt_cache[video].get(f"3{video}{frame}", {})
            out_path = out_dir / run_name / f"SNGS-{video}_{frame}_{run_name}_gt_metrics.png"
            item = render_one(run_name, model, window_size, baseline_kp, line_model, image_path, gt_lines, device, out_path, args.results_root)
            all_summary.append(item)
            print(f"WROTE {run_name} {idx}/{len(rows)} {out_path}", flush=True)
        del model
        torch.cuda.empty_cache()

    with (out_dir / "summary.csv").open("w", newline="") as f:
        fields = [
            "run", "video", "frame", "base_kp", "model_kp", "gt_kp", "line_pred", "gt_lines",
            "base_point", "model_point", "base_line", "model_line", "base_reproj", "model_reproj", "file",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_summary)
    (out_dir / "README.md").write_text(
        "# Standard point/line visual panels\n\n"
        "Top-left: baseline NBJW keypoint heatmap. Cyan x are baseline predicted keypoint peaks; green o are GT keypoints generated from the GT field-line annotation through NBJW KeypointsDB.\n\n"
        "Top-right: temporal/full-finetune model keypoint heatmap. Yellow x are model predicted keypoint peaks; green o are the same GT keypoints.\n\n"
        "Bottom-left: official GT field lines plus GT keypoints.\n\n"
        "Bottom-right: frozen line HRNet heatmap and decoded line endpoints at threshold 0.1. Green lines are GT field lines.\n\n"
        "Small text reports per-frame point/line/reproj and video-level smooth_mean from the corresponding eval results when available.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
