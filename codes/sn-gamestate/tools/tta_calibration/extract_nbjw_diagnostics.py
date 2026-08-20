#!/usr/bin/env python3
"""Small NBJW baseline diagnostics from cached heatmaps.

No TTA and no test GT during frame diagnostics. GT is used only for the final
optional metric summary through the existing evaluator.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from cached_full_test_round2 import _score_job, flatten_params, solve_params, smoothness
from nbjw_calib.utils.utils_heatmap import (
    complete_keypoints,
    coords_to_dict,
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
)


DEFAULT_VIDEOS = [str(x) for x in list(range(116, 151)) + list(range(187, 201))]


def decode(kp_arr, line_arr):
    kp = torch.from_numpy(kp_arr.astype(np.float32)).unsqueeze(0)
    line = torch.from_numpy(line_arr.astype(np.float32)).unsqueeze(0)
    kc = get_keypoints_from_heatmap_batch_maxpool(kp[:, :-1])
    lc = get_keypoints_from_heatmap_batch_maxpool_l(line[:, :-1])
    keypoints = complete_keypoints(
        coords_to_dict(kc, threshold=0.1449),
        coords_to_dict(lc, threshold=0.2983),
        w=960,
        h=540,
        normalize=True,
    )[0]
    return keypoints


def heatmap_score(arr):
    arr = arr[:-1] if arr.shape[0] > 1 else arr
    peaks = arr.reshape(arr.shape[0], -1).max(axis=1)
    active = peaks[peaks > 0]
    return {
        "count": int(active.size),
        "mean": float(active.mean()) if active.size else 0.0,
        "max": float(active.max()) if active.size else 0.0,
    }


def camera_delta(prev, cur):
    a, b = flatten_params(prev), flatten_params(cur)
    if a is None or b is None:
        return ""
    return float(np.linalg.norm(b - a))


def tags(row):
    out = []
    if row["num_keypoints"] < 8:
        out.append("low_kp_count")
    if row["raw_line_score"] < 0.2:
        out.append("low_line_score")
    if row["raw_camera_delta"] not in ("", None) and float(row["raw_camera_delta"]) > 50:
        out.append("large_camera_jump")
    f = row["focal"]
    if f not in ("", None) and (float(f) < 500 or float(f) > 8000):
        out.append("far_zoom")
    return ";".join(out)


def iter_video(cache_root, vid, stride, max_frames):
    files = sorted((Path(cache_root) / "test" / f"SNGS-{vid}").glob("frame_*.npz"))
    used = 0
    for i, path in enumerate(files):
        if i % stride:
            continue
        yield path
        used += 1
        if max_frames and used >= max_frames:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default="outputs/gsr/temporal_hrnet/round2_temporal_calib/cache_hrnet")
    ap.add_argument("--out", default="outputs/tta_calib/baseline_recomputed")
    ap.add_argument("--video-list", nargs="+", default=None)
    ap.add_argument("--max-videos", type=int, default=-1)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--skip-official-eval", action="store_true")
    args = ap.parse_args()

    videos = args.video_list or DEFAULT_VIDEOS
    if args.max_videos and args.max_videos > 0:
        videos = videos[: args.max_videos]

    out = Path(args.out)
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows, params_by_video = [], {}

    for vid in videos:
        prev = None
        params_by_gid = {}
        for path in iter_video(args.cache_root, vid, args.stride, args.max_frames):
            try:
                d = np.load(path)
                frame = int(d["frame"])
                kp_hm, line_hm = d["kp_hm"], d["line_hm"]
                keypoints = decode(kp_hm, line_hm)
                params = solve_params(keypoints)
                gid = f"3{vid}{frame % 1000000:06d}"
                params_by_gid[gid] = params
                kp_s, line_s = heatmap_score(kp_hm), heatmap_score(line_hm)
                row = {
                    "video": vid,
                    "frame": frame,
                    "method": "baseline_raw",
                    "solver_success": bool(params),
                    "use_tta": False,
                    "fallback_reason": "raw",
                    "raw_line_score": line_s["mean"],
                    "tta_line_score": "",
                    "raw_kp_reproj": "",
                    "tta_kp_reproj": "",
                    "raw_camera_delta": camera_delta(prev, params),
                    "tta_camera_delta": "",
                    "num_keypoints": len(keypoints),
                    "num_line_points": line_s["count"],
                    "kp_conf_mean": kp_s["mean"],
                    "kp_conf_max": kp_s["max"],
                    "line_conf_max": line_s["max"],
                    "focal": params.get("x_focal_length", "") if params else "",
                    "k1": "",
                    "pan": params.get("pan_degrees", "") if params else "",
                    "tilt": params.get("tilt_degrees", "") if params else "",
                    "roll": params.get("roll_degrees", "") if params else "",
                }
                row["failure_tags"] = tags(row)
                rows.append(row)
                prev = params
            except Exception as exc:
                rows.append({
                    "video": vid, "frame": Path(path).stem, "method": "baseline_raw",
                    "solver_success": False, "use_tta": False,
                    "fallback_reason": f"error:{type(exc).__name__}", "raw_line_score": "",
                    "tta_line_score": "", "raw_kp_reproj": "", "tta_kp_reproj": "",
                    "raw_camera_delta": "", "tta_camera_delta": "", "num_keypoints": 0,
                    "num_line_points": 0, "kp_conf_mean": "", "kp_conf_max": "",
                    "line_conf_max": "", "focal": "", "k1": "", "pan": "", "tilt": "",
                    "roll": "", "failure_tags": "decode_or_solve_error",
                })
        params_by_video[vid] = params_by_gid
        (pred_dir / f"SNGS-{vid}.json").write_text(json.dumps(params_by_gid, indent=2), encoding="utf-8")

    fields = list(rows[0].keys()) if rows else []
    with (out / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    video_rows = []
    for vid, pb in params_by_video.items():
        vr = [r for r in rows if str(r["video"]) == str(vid)]
        sm = smoothness(pb)
        video_rows.append({
            "video": vid,
            "frames": len(vr),
            "solver_success_rate": sum(bool(r["solver_success"]) for r in vr) / max(1, len(vr)),
            "raw_line_score_mean": float(np.mean([float(r["raw_line_score"]) for r in vr if r["raw_line_score"] != ""])) if vr else 0.0,
            "focal_jitter": sm["mean"],
            "camera_jitter_p95": sm["p95"],
        })
    with (out / "video_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(video_rows[0].keys()) if video_rows else ["video"])
        w.writeheader()
        w.writerows(video_rows)

    summary = {"videos": videos, "n_frames": len(rows), "video_metrics": video_rows}
    if not args.skip_official_eval:
        parts = [_score_job((vid, "baseline_raw", params_by_video[vid])) for vid in videos]
        micro, macro, reproj = [], [], []
        for _name, _vid, mi, ma, rj in parts:
            micro += mi
            macro += ma
            reproj += rj
        summary["official_eval_on_selected_frames"] = {
            "point_acc": float(np.mean(micro)) if micro else None,
            "line_acc": float(np.mean(macro)) if macro else None,
            "reproj_mean": float(np.mean(reproj)) if reproj else None,
            "n_scored_frames": len(micro),
        }
    (out / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Baseline Diagnostics",
        "",
        f"- videos: {', '.join(videos)}",
        f"- frames: {len(rows)}",
        f"- cache_root: {args.cache_root}",
        f"- stride: {args.stride}",
        f"- max_frames_per_video: {args.max_frames or 'all'}",
        "- TTA uses GT: no",
        "- Official metric uses GT: yes, only after predictions are written",
        "",
    ]
    if "official_eval_on_selected_frames" in summary:
        ev = summary["official_eval_on_selected_frames"]
        lines += [
            "## Selected-Frame Official Metrics",
            "",
            f"- point_acc: {ev['point_acc']}",
            f"- line_acc: {ev['line_acc']}",
            f"- reproj_mean: {ev['reproj_mean']}",
            f"- n_scored_frames: {ev['n_scored_frames']}",
            "",
        ]
    lines += [
        "## Outputs",
        "",
        f"- frame_metrics.csv: {out / 'frame_metrics.csv'}",
        f"- video_metrics.csv: {out / 'video_metrics.csv'}",
        f"- predictions/: {pred_dir}",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    reports = Path("outputs/tta_calib/reports")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "baseline_diagnostics.md").write_text((out / "RESULTS.md").read_text(encoding="utf-8"), encoding="utf-8")
    print((out / "RESULTS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
