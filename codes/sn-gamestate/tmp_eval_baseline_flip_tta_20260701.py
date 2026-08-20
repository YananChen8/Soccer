import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import tmp_official_aux_report_eval_visual_20260701 as ref
from nbjw_calib.utils.utils_keypoints import KeypointsDB
from nbjw_calib.utils.utils_linesWC import LineKeypointsWCDB


def _mirror_index(coords, width=105.0, tol=0.2):
    arr = np.asarray(coords, dtype=float)
    out = []
    for x, y in coords:
        target = np.asarray([width - float(x), float(y)], dtype=float)
        d = np.linalg.norm(arr - target[None, :], axis=1)
        j = int(np.argmin(d))
        out.append(j if float(d[j]) <= tol else None)
    if any(i is None for i in out):
        missing = [idx + 1 for idx, v in enumerate(out) if v is None]
        raise RuntimeError(f"missing mirror keypoint mapping: {missing}")
    return out


def keypoint_swap():
    db = KeypointsDB({}, torch.zeros(3, 540, 960))
    # NBJW keypoint HRNet emits 57 semantic keypoints + 1 extra/background channel.
    return _mirror_index(db.keypoint_world_coords_2D) + [57]


def line_swap():
    obj = LineKeypointsWCDB(Image.new("RGB", (960, 540)), np.eye(3), (960, 540))
    names = [n.strip() for n in obj.lines_list]
    name_to_idx = {n: i for i, n in enumerate(names)}
    manual = {
        "Goal left crossbar": "Goal right crossbar",
        "Goal right crossbar": "Goal left crossbar",
        "Goal left post left": "Goal right post right",
        "Goal right post right": "Goal left post left",
        "Goal left post right": "Goal right post left",
        "Goal right post left": "Goal left post right",
        "Side line left": "Side line right",
        "Side line right": "Side line left",
    }

    def mirror_name(name):
        if name in manual:
            return manual[name]
        if ". left " in name:
            return name.replace(". left ", ". right ")
        if ". right " in name:
            return name.replace(". right ", ". left ")
        return name

    out = [name_to_idx[mirror_name(n)] for n in names]
    # NBJW line HRNet emits 23 semantic line channels + 1 extra/background channel.
    return out + [23]


def align_flipped_heatmap(hm, swap):
    h = torch.flip(hm, dims=[-1])
    if h.shape[1] != len(swap):
        raise RuntimeError(f"heatmap channels={h.shape[1]} swap={len(swap)}")
    return h[:, swap, :, :]


def local_peak_refine_heatmap(hm, radius=2, temperature=0.03):
    refined = hm.clone()
    bsz, channels, height, width = hm.shape
    for b in range(bsz):
        for c in range(channels - 1):
            flat_idx = int(torch.argmax(hm[b, c]).item())
            x0 = flat_idx % width
            y0 = flat_idx // width
            x1, x2 = max(0, x0 - radius), min(width, x0 + radius + 1)
            y1, y2 = max(0, y0 - radius), min(height, y0 + radius + 1)
            patch = hm[b, c, y1:y2, x1:x2].float()
            weights = torch.softmax((patch - patch.max()).reshape(-1) / max(temperature, 1e-6), dim=0).reshape_as(patch)
            yy, xx = torch.meshgrid(
                torch.arange(y1, y2, device=hm.device, dtype=torch.float32),
                torch.arange(x1, x2, device=hm.device, dtype=torch.float32),
                indexing="ij",
            )
            xr = int(torch.round((weights * xx).sum()).clamp(0, width - 1).item())
            yr = int(torch.round((weights * yy).sum()).clamp(0, height - 1).item())
            if xr != x0 or yr != y0:
                peak = refined[b, c, y0, x0].clone()
                refined[b, c, y0, x0] = torch.minimum(refined[b, c, y0, x0], peak * 0.999)
                refined[b, c, yr, xr] = torch.maximum(refined[b, c, yr, xr], peak)
    return refined


def evaluate(args):
    device = torch.device(args.device)
    frames_root, data_root = ref.get_split_paths("test")
    videos = [str(v).replace("SNGS-", "") for v in args.videos]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kp_swap = keypoint_swap()
    ln_swap = line_swap()
    (out_dir / "flip_mappings.json").write_text(json.dumps({"keypoint_swap": kp_swap, "line_swap": ln_swap}, indent=2))

    kp_model, line_model = ref.base.load_hrnets(device)
    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    mode_set = set(args.modes)
    modes = ["baseline_raw"]
    if "flip" in mode_set:
        modes.append("baseline_flip_tta")
    if "local_peak" in mode_set:
        modes.append("baseline_local_peak_refine")
    if "flip_local_peak" in mode_set:
        modes.append("baseline_flip_local_peak")
    rows_by_mode = {m: [] for m in modes}
    frame_rows = []

    for video in videos:
        files = sorted((frames_root / f"SNGS-{video}" / "img1").glob("*.jpg"))
        gt = ref.base.load_gt_lines_for_video(str(data_root), video)
        id_map = ref.image_id_map(data_root, video)
        start = time.perf_counter()
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
                kp_flip = align_flipped_heatmap(kp_model(flip_img), kp_swap)
                line_flip = align_flipped_heatmap(line_model(flip_img), ln_swap)
            outputs = {"baseline_raw": (kp_raw, line_raw)}
            if "flip" in mode_set:
                outputs["baseline_flip_tta"] = ((kp_raw + kp_flip) * 0.5, (line_raw + line_flip) * 0.5)
            if "local_peak" in mode_set:
                outputs["baseline_local_peak_refine"] = (
                    local_peak_refine_heatmap(kp_raw, args.peak_radius, args.peak_temperature),
                    local_peak_refine_heatmap(line_raw, args.peak_radius, args.peak_temperature),
                )
            if "flip_local_peak" in mode_set:
                kp_fused = (kp_raw + kp_flip) * 0.5
                line_fused = (line_raw + line_flip) * 0.5
                outputs["baseline_flip_local_peak"] = (
                    local_peak_refine_heatmap(kp_fused, args.peak_radius, args.peak_temperature),
                    local_peak_refine_heatmap(line_fused, args.peak_radius, args.peak_temperature),
                )
            for mode, (kp_hm, line_hm) in outputs.items():
                s = ref.score_hm(kp_hm, line_hm, gt[gid])
                row = {
                    "run": mode,
                    "split": "test",
                    "video": video,
                    "frame": image_path.stem,
                    "image_path": str(image_path),
                    "point_acc": s["point_acc"],
                    "line_acc": s["line_acc"],
                    "reproj_mean": s["reproj_mean"],
                    "reproj": s["reproj"],
                    "params": s["params"],
                }
                rows_by_mode[mode].append(row)
                frame_rows.append({k: v for k, v in row.items() if k not in ("params", "reproj")})
        print(f"video={video} seconds={time.perf_counter() - start:.1f}", flush=True)

    results = {}
    for mode, rows in rows_by_mode.items():
        results[mode] = {"videos": {}, "aggregate": None}
        for video in videos:
            vr = [r for r in rows if r["video"] == video]
            results[mode]["videos"][video] = ref.summarize(vr)
        vids = results[mode]["videos"]
        results[mode]["aggregate"] = {
            "point_acc": ref.mean([v["point_acc"] for v in vids.values()]),
            "line_acc": ref.mean([v["line_acc"] for v in vids.values()]),
            "reproj_mean": ref.mean([v["reproj_mean"] for v in vids.values()]),
            "smooth_mean": ref.mean([v["smooth_mean"] for v in vids.values()]),
            "JaC@5": ref.mean([v["JaC@5"] for v in vids.values()]),
            "JaC@10": ref.mean([v["JaC@10"] for v in vids.values()]),
            "JaC@15": ref.mean([v["JaC@15"] for v in vids.values()]),
            "JaC@20": ref.mean([v["JaC@20"] for v in vids.values()]),
            "MRE": ref.mean([v["MRE"] for v in vids.values()]),
            "CR": ref.mean([v["CR"] for v in vids.values()]),
            "Final Score": ref.mean([v["Final Score"] for v in vids.values()]),
            "camera_smooth_l2_mean": ref.mean([v["camera_smooth_l2_mean"] for v in vids.values()]),
            "camera_smooth_l2_p95": ref.mean([v["camera_smooth_l2_p95"] for v in vids.values()]),
            "n_total": int(sum(v["n_total"] for v in vids.values())),
            "n_scored": int(sum(v["n_scored"] for v in vids.values())),
        }

    (out_dir / "test_results.json").write_text(json.dumps(results, indent=2))
    with (out_dir / "test_frame_scores.csv").open("w", newline="") as f:
        fields = ["run", "split", "video", "frame", "image_path", "point_acc", "line_acc", "reproj_mean"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(frame_rows)
    ref.write_tables(results, out_dir, "test")
    write_delta_report(results, out_dir, args)


def write_delta_report(results, out_dir, args):
    raw = results["baseline_raw"]["aggregate"]
    keys = ["point_acc", "line_acc", "reproj_mean", "JaC@5", "JaC@10", "JaC@15", "JaC@20", "MRE", "CR", "Final Score", "camera_smooth_l2_mean", "camera_smooth_l2_p95"]
    lines = [
        "# Baseline TTA Evaluation",
        "",
        f"- videos: {' '.join(args.videos)}",
        f"- stride: {args.stride}",
        "- flip-TTA: horizontal flip, semantic channel swap, average keypoint and line heatmaps, original NBJW decode/solver.",
        f"- local-peak refinement: radius={args.peak_radius}, temperature={args.peak_temperature}, keypoint and line heatmap peaks refined before original NBJW decode/solver.",
        "- Reference evaluator was not modified; this wrapper reuses its scoring/summarization functions.",
        "",
    ]
    for mode in [m for m in results if m != "baseline_raw"]:
        tta = results[mode]["aggregate"]
        lines += [
            f"## {mode}",
            "",
            "| metric | raw | tta | delta |",
            "|---|---:|---:|---:|",
        ]
        for k in keys:
            rv, tv = raw.get(k), tta.get(k)
            delta = None if rv is None or tv is None else tv - rv
            lines.append(f"| {k} | {ref.fmt(rv, 6 if k not in ('MRE','reproj_mean','camera_smooth_l2_mean','camera_smooth_l2_p95') else 3)} | {ref.fmt(tv, 6 if k not in ('MRE','reproj_mean','camera_smooth_l2_mean','camera_smooth_l2_p95') else 3)} | {ref.fmt(delta, 6 if k not in ('MRE','reproj_mean','camera_smooth_l2_mean','camera_smooth_l2_p95') else 3)} |")
        lines.append("")
    lines.append("")
    (out_dir / "baseline_tta_report.md").write_text("\n".join(lines) + "\n")
    if "baseline_flip_tta" in results:
        (out_dir / "baseline_flip_tta_report.md").write_text("\n".join(lines) + "\n")
    if "baseline_local_peak_refine" in results:
        (out_dir / "baseline_local_peak_refine_report.md").write_text("\n".join(lines) + "\n")
    if "baseline_flip_local_peak" in results:
        (out_dir / "baseline_flip_local_peak_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["116", "117", "118", "119", "120", "121", "122", "123"])
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--modes", nargs="+", choices=["flip", "local_peak", "flip_local_peak"], default=["flip"])
    ap.add_argument("--peak-radius", type=int, default=2)
    ap.add_argument("--peak-temperature", type=float, default=0.03)
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
