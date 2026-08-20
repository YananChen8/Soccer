import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

import tmp_eval_baseline_flip_tta_20260701 as tta
import tmp_official_aux_report_eval_visual_20260701 as ref
from nbjw_calib.utils.utils_keypoints import KeypointsDB


def peak_xy(hm):
    flat = hm.flatten(2)
    idx = flat.argmax(dim=-1)
    h, w = hm.shape[-2:]
    x = (idx % w).float()
    y = (idx // w).float()
    return torch.stack([x, y], dim=-1), flat.amax(dim=-1)


def norm_heat(heat):
    arr = heat.detach().float().cpu().numpy()
    lo, hi = np.percentile(arr, [1, 99.5])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)


def heat_overlay(image, heat, color):
    h = Image.fromarray((norm_heat(heat) * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    rgba = Image.new("RGBA", image.size, color + (0,))
    rgba.putalpha(h.point(lambda v: int(v * 0.65)))
    return Image.alpha_composite(image.convert("RGBA"), rgba).convert("RGB")


def draw_gt_lines(image, lines, color=(60, 255, 80), width=2):
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


def draw_peaks(image, hm, color, radius=4):
    draw = ImageDraw.Draw(image)
    xy, conf = peak_xy(hm[:, :-1])
    n = 0
    for x, y, s in zip(xy[0, :, 0], xy[0, :, 1], conf[0]):
        if float(s) <= 0:
            continue
        px, py = float(x) * 2.0, float(y) * 2.0
        draw.line((px - radius, py, px + radius, py), fill=color, width=2)
        draw.line((px, py - radius, px, py + radius), fill=color, width=2)
        n += 1
    return n


def draw_pred_lines(image, params, color, width=2):
    try:
        polys = ref.base.get_polylines(params, ref.base.WIDTH, ref.base.HEIGHT, sampling_factor=0.9)
    except Exception:
        return
    draw = ImageDraw.Draw(image)
    for line in polys.values():
        xy = []
        for p in line:
            try:
                if isinstance(p, dict):
                    x, y = float(p["x"]), float(p["y"])
                else:
                    x, y = float(p[0]), float(p[1])
                if 0 <= x <= 1.5 and 0 <= y <= 1.5:
                    x *= image.size[0]
                    y *= image.size[1]
                else:
                    x /= 2.0
                    y /= 2.0
                xy.append((x, y))
            except Exception:
                pass
        if len(xy) >= 2:
            draw.line(xy, fill=color, width=width)


def label(image, text):
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.size[0], 30), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))


def concat(panels):
    w, h = panels[0].size
    canvas = Image.new("RGB", (w * 2, h * 2), (20, 20, 20))
    for i, p in enumerate(panels):
        canvas.paste(p, ((i % 2) * w, (i // 2) * h))
    return canvas


def gt_heatmap(gt_lines, img_tensor):
    gt, mask = KeypointsDB(gt_lines or {}, img_tensor.cpu()[0]).get_tensor_w_mask()
    gt = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0)
    mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
    return gt, mask


def update_channel_stats(stats, mode, video, hm, raw_hm=None):
    pred_xy, conf = peak_xy(hm[:, :-1].detach().cpu())
    raw_xy = None
    if raw_hm is not None:
        raw_xy, _ = peak_xy(raw_hm[:, :-1].detach().cpu())
    for c in range(conf.shape[1]):
        key = (mode, str(video), int(c))
        item = stats.setdefault(key, {"n": 0, "conf": [], "move": []})
        item["n"] += 1
        item["conf"].append(float(conf[0, c]))
        if raw_xy is not None:
            item["move"].append(float(torch.linalg.norm((pred_xy[0, c] - raw_xy[0, c]) * 2.0)))


def make_outputs(kp_raw, line_raw, kp_flip, line_flip, radius, temp):
    kp_fused = (kp_raw + kp_flip) * 0.5
    line_fused = (line_raw + line_flip) * 0.5
    return {
        "raw": (kp_raw, line_raw),
        "flip": (kp_fused, line_fused),
        "local_peak": (
            tta.local_peak_refine_heatmap(kp_raw, radius, temp),
            tta.local_peak_refine_heatmap(line_raw, radius, temp),
        ),
        "flip_local_peak": (
            tta.local_peak_refine_heatmap(kp_fused, radius, temp),
            tta.local_peak_refine_heatmap(line_fused, radius, temp),
        ),
    }


def read_frame_scores(path):
    rows = list(csv.DictReader(open(path, newline="")))
    by = {}
    for r in rows:
        by[(r["video"], r["frame"], r["run"])] = r
    return by


def choose_cases(score_path, method, metric="point_acc", n=3):
    by = read_frame_scores(score_path)
    rows = []
    run = {
        "flip": "baseline_flip_tta",
        "local_peak": "baseline_local_peak_refine",
        "flip_local_peak": "baseline_flip_local_peak",
    }[method]
    keys = sorted({(v, f) for v, f, _ in by})
    for v, f in keys:
        raw = by.get((v, f, "baseline_raw"))
        cur = by.get((v, f, run))
        if not raw or not cur:
            continue
        try:
            d = float(cur[metric]) - float(raw[metric])
            rows.append((d, v, f))
        except Exception:
            pass
    return sorted(rows, reverse=True)[:n]


def render_case(out_path, image_path, gt_lines, outputs, scores, method):
    image = Image.open(image_path).convert("RGB").resize((960, 540))
    raw_kp, raw_line = outputs["raw"]
    cur_kp, cur_line = outputs[method]
    raw_s = scores["raw"]
    cur_s = scores[method]

    p1 = heat_overlay(image, raw_kp[0, :-1].max(0).values, (255, 0, 0))
    draw_gt_lines(p1, gt_lines, width=1)
    draw_peaks(p1, raw_kp, (0, 255, 255))
    label(p1, f"raw kp heat/peaks p={raw_s['point_acc']:.3f} l={raw_s['line_acc']:.3f}")

    p2 = heat_overlay(image, cur_kp[0, :-1].max(0).values, (255, 0, 0))
    draw_gt_lines(p2, gt_lines, width=1)
    draw_peaks(p2, cur_kp, (255, 230, 0))
    label(p2, f"{method} kp heat/peaks p={cur_s['point_acc']:.3f} l={cur_s['line_acc']:.3f}")

    p3 = image.copy()
    draw_gt_lines(p3, gt_lines, width=2)
    draw_pred_lines(p3, raw_s["params"], (0, 255, 255), width=2)
    label(p3, f"raw calib cyan reproj={raw_s['reproj_mean']:.2f}")

    p4 = image.copy()
    draw_gt_lines(p4, gt_lines, width=2)
    draw_pred_lines(p4, cur_s["params"], (255, 230, 0), width=2)
    label(p4, f"{method} calib yellow reproj={cur_s['reproj_mean']:.2f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat([p1, p2, p3, p4]).save(out_path, quality=92)


def plot_channel_stats(rows, out_dir):
    coords = KeypointsDB({}, torch.zeros(3, 540, 960)).keypoint_world_coords_2D
    x_by_channel = {i: float(coords[i][0]) for i in range(len(coords))}
    with (out_dir / "keypoint_channel_hit5_by_video.csv").open("w", newline="") as f:
        fields = ["mode", "video", "channel", "field_x", "n", "mean_peak_conf", "mean_move_px_from_raw"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (mode, video, channel), v in sorted(rows.items()):
            w.writerow({
                "mode": mode,
                "video": video,
                "channel": channel + 1,
                "field_x": x_by_channel[channel],
                "n": v["n"],
                "mean_peak_conf": float(np.mean(v["conf"])) if v["conf"] else "",
                "mean_move_px_from_raw": float(np.mean(v["move"])) if v["move"] else 0.0,
            })

    def collect(modes):
        xs, ys, cs = [], [], []
        for (mode, video, channel), v in rows.items():
            if mode not in modes:
                continue
            xs.append(x_by_channel[channel] + (0.45 if mode != "raw" else -0.45))
            ys.append(float(np.mean(v["conf"])) if v["conf"] else 0.0)
            cs.append(mode)
        return xs, ys, cs

    color = {"raw": "#1f77b4", "flip": "#ff7f0e", "local_peak": "#2ca02c", "flip_local_peak": "#d62728"}
    for name, modes, title in [
        ("flip_channel_hit5_scatter.png", ["raw", "flip"], "Flip TTA: keypoint channel peak confidence by field x"),
        ("local_peak_channel_hit5_scatter.png", ["raw", "local_peak"], "Local peak refinement: keypoint channel peak confidence by field x"),
        ("combined_channel_hit5_scatter.png", ["raw", "flip_local_peak"], "Flip + local peak: keypoint channel peak confidence by field x"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5))
        xs, ys, cs = collect(modes)
        for mode in modes:
            xx = [x for x, c in zip(xs, cs) if c == mode]
            yy = [y for y, c in zip(ys, cs) if c == mode]
            ax.scatter(xx, yy, s=22, alpha=0.65, label=mode, color=color[mode])
        ax.set_xlabel("semantic keypoint field x position, meters (left -> right)")
        ax.set_ylabel("mean peak confidence per video/channel")
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for mode in ["flip", "local_peak", "flip_local_peak"]:
        xx, yy = [], []
        for (m, _video, channel), v in rows.items():
            if m == mode and v["move"]:
                xx.append(x_by_channel[channel])
                yy.append(float(np.mean(v["move"])))
        ax.scatter(xx, yy, s=22, alpha=0.55, label=mode, color=color[mode])
    ax.set_xlabel("semantic keypoint field x position, meters (left -> right)")
    ax.set_ylabel("mean peak movement from raw, image px")
    ax.set_title("How much each TTA moves semantic keypoint peaks")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "keypoint_channel_peak_movement_from_raw.png", dpi=160)
    plt.close(fig)


def plot_frame_deltas(score_path, out_dir):
    rows = list(csv.DictReader(open(score_path, newline="")))
    by = {(r["video"], r["frame"], r["run"]): r for r in rows}
    data = []
    for v, f in sorted({(r["video"], r["frame"]) for r in rows}):
        raw = by.get((v, f, "baseline_raw"))
        if not raw:
            continue
        for run, label in [
            ("baseline_flip_tta", "flip"),
            ("baseline_local_peak_refine", "local_peak"),
            ("baseline_flip_local_peak", "flip_local_peak"),
        ]:
            cur = by.get((v, f, run))
            if cur:
                try:
                    data.append({
                        "method": label,
                        "d_point": float(cur["point_acc"]) - float(raw["point_acc"]),
                        "d_line": float(cur["line_acc"]) - float(raw["line_acc"]),
                        "d_reproj": float(cur["reproj_mean"]) - float(raw["reproj_mean"]),
                    })
                except Exception:
                    pass
    with (out_dir / "frame_delta_summary.csv").open("w", newline="") as f:
        fields = ["method", "d_point", "d_line", "d_reproj"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    colors = {"flip": "#ff7f0e", "local_peak": "#2ca02c", "flip_local_peak": "#d62728"}
    for ax, metric, title in zip(axs, ["d_point", "d_line", "d_reproj"], ["delta point_acc", "delta line_acc", "delta reproj"]):
        vals = [[r[metric] for r in data if r["method"] == m] for m in colors]
        ax.boxplot(vals, labels=list(colors), showfliers=False)
        for i, m in enumerate(colors, 1):
            yy = [r[metric] for r in data if r["method"] == m]
            xx = np.random.default_rng(42 + i).normal(i, 0.035, size=len(yy))
            ax.scatter(xx, yy, s=8, alpha=0.25, color=colors[m])
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "observation_tta_frame_delta_distributions.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["116", "117", "118", "119", "120", "121", "122", "123"])
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--results-dir", default="outputs/tta_calib/baseline_flip_tta_20260701/test_116_123_s20")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--peak-radius", type=int, default=2)
    ap.add_argument("--peak-temperature", type=float, default=0.03)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    score_path = results_dir / "test_frame_scores.csv"
    plot_frame_deltas(score_path, out_dir)

    device = torch.device(args.device)
    frames_root, data_root = ref.get_split_paths("test")
    kp_model, line_model = ref.base.load_hrnets(device)
    kp_swap = tta.keypoint_swap()
    ln_swap = tta.line_swap()
    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    stats = {}
    case_map = {}
    for method in ["flip", "local_peak", "flip_local_peak"]:
        case_map[method] = {(v, f): d for d, v, f in choose_cases(score_path, method, n=3)}

    for video in args.videos:
        files = sorted((frames_root / f"SNGS-{video}" / "img1").glob("*.jpg"))
        gt = ref.base.load_gt_lines_for_video(str(data_root), video)
        id_map = ref.image_id_map(data_root, video)
        for idx, image_path in enumerate(files):
            if idx % args.stride != 0:
                continue
            gid = id_map.get(image_path.stem, f"3{video}{image_path.stem}")
            if gid not in gt:
                continue
            pil = Image.open(image_path).convert("RGB")
            img = tfm(pil).unsqueeze(0).to(device)
            flip_img = torch.flip(img, dims=[-1])
            with torch.no_grad():
                kp_raw = kp_model(img)
                line_raw = line_model(img)
                kp_flip = tta.align_flipped_heatmap(kp_model(flip_img), kp_swap)
                line_flip = tta.align_flipped_heatmap(line_model(flip_img), ln_swap)
            outputs = make_outputs(kp_raw, line_raw, kp_flip, line_flip, args.peak_radius, args.peak_temperature)
            for mode, (kp_hm, _line_hm) in outputs.items():
                update_channel_stats(stats, mode, video, kp_hm, outputs["raw"][0] if mode != "raw" else None)
            needed = [m for m, cases in case_map.items() if (video, image_path.stem) in cases]
            if needed:
                scores = {m: ref.score_hm(kp, line, gt[gid]) for m, (kp, line) in outputs.items()}
                for method in needed:
                    delta = case_map[method][(video, image_path.stem)]
                    render_case(
                        out_dir / "example_frames" / method / f"SNGS-{video}_{image_path.stem}_{method}_dpoint_{delta:+.3f}.jpg",
                        image_path, gt[gid], outputs, scores, method
                    )
        print(f"done video {video}", flush=True)

    plot_channel_stats(stats, out_dir)
    report = [
        "# TTA Presentation Visuals",
        "",
        "- `flip_channel_hit5_scatter.png`: raw vs flip keypoint channel peak confidence by field x position. This is a detector proxy, not official point_acc.",
        "- `local_peak_channel_hit5_scatter.png`: raw vs local peak keypoint channel peak confidence.",
        "- `combined_channel_hit5_scatter.png`: raw vs flip+local peak keypoint channel peak confidence.",
        "- `keypoint_channel_peak_movement_from_raw.png`: how much each TTA moves semantic channel peaks from raw.",
        "- `observation_tta_frame_delta_distributions.png`: per-frame delta distributions.",
        "- `example_frames/`: selected top-positive point_acc frames with raw vs TTA heatmaps and calibration overlays.",
        "",
        "Local peak refinement explanation: for each semantic heatmap channel, take the current argmax, compute a soft-argmax only inside a radius-2 local window, then move the discrete peak to that local soft coordinate. It changes observations, not camera parameters.",
    ]
    (out_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
