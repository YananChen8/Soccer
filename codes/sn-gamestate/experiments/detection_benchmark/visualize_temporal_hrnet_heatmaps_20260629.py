import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from experiments.detection_benchmark import eval_temporal_input_mixer_calib as base
from experiments.detection_benchmark.eval_temporal_feature_fusion_calib import load_feature_model


RUNS = [
    ("baseline", None),
    ("k10_smallres_plain", "feature_fusion_last_smallres_k10_train12/checkpoints/feature_fusion_last_smallres_k10_train12_k10_ep3.pt"),
    ("k10_lora_plain", "feature_fusion_last_lora_k10_train12/checkpoints/feature_fusion_last_lora_k10_train12_k10_ep3.pt"),
    ("mass5_smallres", "feature_fusion_mass5_k10_smallres_train12/checkpoints/feature_fusion_mass5_k10_smallres_train12_k10_ep3.pt"),
    ("mass5_lora", "feature_fusion_mass5_k10_lora_train12/checkpoints/feature_fusion_mass5_k10_lora_train12_k10_ep3.pt"),
    ("peak_sharp_smallres", "feature_fusion_peak_sharp_k10_smallres_train12/checkpoints/feature_fusion_peak_sharp_k10_smallres_train12_k10_ep3.pt"),
    ("peak_sharp_lora", "feature_fusion_peak_sharp_k10_lora_train12/checkpoints/feature_fusion_peak_sharp_k10_lora_train12_k10_ep3.pt"),
    ("peak_sharp_msehigh", "feature_fusion_peak_sharp_msehigh_k10_smallres_train12/checkpoints/feature_fusion_peak_sharp_msehigh_k10_smallres_train12_k10_ep3.pt"),
]

HUB = Path("outputs/gsr/temporal_hrnet/temporal_calib_results_hub")


def norm_heatmap(hm):
    arr = hm.detach().float().cpu().numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr, [1, 99.5])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return arr


def heat_overlay(image, heat, color):
    heat_img = Image.fromarray((norm_heatmap(heat) * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    rgba = Image.new("RGBA", image.size, color + (0,))
    rgba.putalpha(heat_img.point(lambda v: int(v * 0.75)))
    return Image.alpha_composite(image.convert("RGBA"), rgba).convert("RGB")


def draw_lines(image, lines, color, width=2):
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


def draw_pred_lines(image, params, color=(0, 255, 255), width=2):
    try:
        pred = base.get_polylines(params, base.WIDTH, base.HEIGHT, sampling_factor=0.9)
    except Exception:
        return
    draw_lines(image, pred, color, width=width)


def xy_from_value(value):
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return value["x"], value["y"]
        if "point" in value:
            return xy_from_value(value["point"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return value[0], value[1]
    return None


def draw_keypoints(image, keypoints, color=(255, 0, 255)):
    draw = ImageDraw.Draw(image)
    count = 0
    for _name, value in (keypoints or {}).items():
        xy = xy_from_value(value)
        if xy is None:
            continue
        try:
            x, y = float(xy[0]), float(xy[1])
        except Exception:
            continue
        if 0 <= x <= 1.5 and 0 <= y <= 1.5:
            x *= image.size[0]
            y *= image.size[1]
        else:
            x /= 2.0
            y /= 2.0
        if np.isfinite(x) and np.isfinite(y):
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
            count += 1
    return count


def label_panel(image, text):
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.size[0], 24), fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 255, 255))


def choose_frames(video, gt, frames_per_video):
    files = sorted(Path(base.FRAMES, f"SNGS-{video}", "img1").glob("*.jpg"))
    valid = [p for p in files if f"3{video}{p.stem}" in gt]
    if not valid:
        return []
    center = len(valid) // 2
    starts = list(range(center, len(valid) - frames_per_video + 1)) + list(range(0, center))
    for start in starts:
        chunk = valid[start : start + frames_per_video]
        nums = [int(p.stem) for p in chunk if p.stem.isdigit()]
        if len(nums) == frames_per_video and all(b == a + 1 for a, b in zip(nums, nums[1:])):
            return chunk
    return valid[center : center + frames_per_video]


def build_window(image_path, window_size, tfm):
    files = sorted(image_path.parent.glob("*.jpg"))
    idx = files.index(image_path)
    start = max(0, idx - window_size + 1)
    frames = [tfm(Image.open(p).convert("RGB")) for p in files[start : idx + 1]]
    return base.left_pad_window(frames, window_size)


def concat_grid(panels):
    w, h = panels[0].size
    canvas = Image.new("RGB", (w * 2, h * 2), (20, 20, 20))
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % 2) * w, (i // 2) * h))
    return canvas


def render_one(run_name, model, window_size, baseline_kp, line_model, image_path, gt_lines, device, out_path):
    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    image = Image.open(image_path).convert("RGB").resize((960, 540))
    x = tfm(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        line_hm = line_model(x)
        if model is None:
            kp_hm = baseline_kp(x)
        else:
            win = build_window(image_path, window_size, tfm).unsqueeze(0).to(device)
            kp_hm = model(win)
    keypoints = base.decode_keypoints(kp_hm, line_hm)
    params = base.solve_params(keypoints)

    gt_panel = image.copy()
    draw_lines(gt_panel, gt_lines, (0, 255, 0), width=2)
    label_panel(gt_panel, "GT field lines on image")

    point_heat = kp_hm[0, :-1].max(0).values
    point_panel = heat_overlay(image, point_heat, (255, 0, 0))
    draw_lines(point_panel, gt_lines, (0, 255, 0), width=1)
    label_panel(point_panel, "keypoint heatmap max, GT lines")

    line_heat = line_hm[0, :-1].max(0).values
    line_panel = heat_overlay(image, line_heat, (0, 128, 255))
    draw_lines(line_panel, gt_lines, (0, 255, 0), width=1)
    label_panel(line_panel, "line heatmap max, GT lines")

    pred_panel = image.copy()
    draw_lines(pred_panel, gt_lines, (0, 255, 0), width=2)
    draw_pred_lines(pred_panel, params, (0, 255, 255), width=2)
    kp_count = draw_keypoints(pred_panel, keypoints, (255, 0, 255))
    label_panel(pred_panel, f"pred points {kp_count}, pred lines cyan, GT green")

    grid = concat_grid([gt_panel, point_panel, line_panel, pred_panel])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, quality=92)
    return {
        "run": run_name,
        "frame": image_path.stem,
        "file": str(out_path),
        "keypoints": kp_count,
        "solved": bool(params),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["116", "117", "118", "119", "120", "121", "122", "123"])
    ap.add_argument("--frames-per-video", type=int, default=3)
    ap.add_argument("--out-dir", default=str(HUB / "teacher_report_20260628_feature_fusion" / "heatmap_visuals_20260629"))
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_kp, line_model = base.load_hrnets(device)
    index_rows = []
    selected = {}
    runs = RUNS[: args.max_runs] if args.max_runs else RUNS
    for run_name, rel_ckpt in runs:
        print("RUN", run_name, flush=True)
        model, window_size = None, 1
        if rel_ckpt is not None:
            kp_for_model, _ = base.load_hrnets(device)
            model, window_size, _ck = load_feature_model(str(HUB / rel_ckpt), kp_for_model, device)
        for video in args.videos:
            gt = base.load_gt_lines_for_video(base.DATA_ROOT, video)
            if video not in selected:
                selected[video] = [str(p) for p in choose_frames(video, gt, args.frames_per_video)]
            for image_path_s in selected[video]:
                image_path = Path(image_path_s)
                gid = f"3{video}{image_path.stem}"
                gt_lines = gt.get(gid, {})
                rel = Path(run_name) / f"SNGS-{video}" / f"{image_path.stem}.jpg"
                row = render_one(
                    run_name, model, window_size, baseline_kp, line_model,
                    image_path, gt_lines, device, out_dir / rel
                )
                row["video"] = video
                row["image_path"] = str(image_path)
                index_rows.append(row)
        del model
        torch.cuda.empty_cache()

    with (out_dir / "selected_frames.json").open("w") as f:
        json.dump(selected, f, indent=2)
    with (out_dir / "index.csv").open("w", newline="") as f:
        fields = ["run", "video", "frame", "keypoints", "solved", "file", "image_path"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index_rows)
    print("WROTE", out_dir, "images", len(index_rows), flush=True)


if __name__ == "__main__":
    main()
