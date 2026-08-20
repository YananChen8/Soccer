import argparse
import csv
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
    rgba.putalpha(heat_img.point(lambda v: int(v * 0.70)))
    return Image.alpha_composite(image.convert("RGBA"), rgba).convert("RGB")


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


def label(image, text):
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.size[0], 28), fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255))


def draw_cross(draw, x, y, color):
    r = 5
    draw.line((x - r, y, x + r, y), fill=color, width=2)
    draw.line((x, y - r, x, y + r), fill=color, width=2)


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
        draw.text((x2 + 6, y2 - 8), str(channel), fill=(255, 230, 0))
    return len(lines)


def build_window(image_path, window_size, tfm):
    files = sorted(image_path.parent.glob("*.jpg"))
    idx = files.index(image_path)
    start = max(0, idx - window_size + 1)
    frames = [tfm(Image.open(p).convert("RGB")) for p in files[start : idx + 1]]
    return base.left_pad_window(frames, window_size)


def render_one(run_name, model, window_size, baseline_kp, line_model, image_path, gt_lines, device, out_path):
    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    image = Image.open(image_path).convert("RGB").resize((960, 540))
    x = tfm(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    win = build_window(image_path, window_size, tfm).unsqueeze(0).to(device)
    with torch.no_grad():
        line_hm = line_model(x)
        base_hm = baseline_kp(x)
        fused_hm = model(win)

    base_panel = heat_overlay(image, base_hm[0, :-1].max(0).values, (255, 0, 0))
    draw_gt_lines(base_panel, gt_lines, width=2)
    n_base = draw_kp_peaks(base_panel, base_hm, color=(0, 255, 255))
    label(base_panel, f"baseline keypoints n={n_base}; GT lines green")

    fused_panel = heat_overlay(image, fused_hm[0, :-1].max(0).values, (255, 0, 0))
    draw_gt_lines(fused_panel, gt_lines, width=2)
    n_fused = draw_kp_peaks(fused_panel, fused_hm, color=(255, 230, 0))
    label(fused_panel, f"{run_name} keypoints n={n_fused}; GT lines green")

    gt_panel = image.copy()
    draw_gt_lines(gt_panel, gt_lines, width=3)
    label(gt_panel, f"official GT field lines n={len(gt_lines or {})}")

    line_panel = heat_overlay(image, line_hm[0, :-1].max(0).values, (0, 128, 255))
    draw_gt_lines(line_panel, gt_lines, width=1)
    n_line = draw_line_endpoints(line_panel, line_hm, threshold=0.1)
    label(line_panel, f"frozen line HRNet endpoints @0.1 n={n_line}; GT green")

    canvas = Image.new("RGB", (1920, 1080), (20, 20, 20))
    canvas.paste(base_panel, (0, 0))
    canvas.paste(fused_panel, (960, 0))
    canvas.paste(gt_panel, (0, 540))
    canvas.paste(line_panel, (960, 540))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    return {
        "run": run_name,
        "video": image_path.parent.parent.name.replace("SNGS-", ""),
        "frame": image_path.stem,
        "base_kp": n_base,
        "model_kp": n_fused,
        "line_pred": n_line,
        "gt_lines": len(gt_lines or {}),
        "file": str(out_path),
    }


def read_frames(path, limit):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-csv", required=True)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    baseline_kp, line_model = base.load_hrnets(device)
    rows = read_frames(args.frame_csv, args.limit)
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
            out_path = out_dir / run_name / f"SNGS-{video}_{frame}_{run_name}_points_lines.png"
            item = render_one(run_name, model, window_size, baseline_kp, line_model, image_path, gt_lines, device, out_path)
            all_summary.append(item)
            print(f"WROTE {run_name} {idx}/{len(rows)} {out_path}", flush=True)
        del model
        torch.cuda.empty_cache()

    with (out_dir / "summary.csv").open("w", newline="") as f:
        fields = ["run", "video", "frame", "base_kp", "model_kp", "line_pred", "gt_lines", "file"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_summary)
    (out_dir / "README.md").write_text(
        "# Lossfix Full-Finetune Point/Line Visuals\n\n"
        "Per image: top-left baseline keypoint heatmap/peaks, top-right model keypoint heatmap/peaks, "
        "bottom-left official GT field lines, bottom-right frozen line HRNet heatmap and endpoints at threshold 0.1. "
        "Green lines are official GT field lines. Cyan crosses are baseline keypoint peaks. Yellow crosses/segments are model keypoint peaks or line endpoints.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
