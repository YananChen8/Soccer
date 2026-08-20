#!/usr/bin/env python3
"""Fast camera-level TTA probe from existing NBJW camera params.

This avoids the slow NBJW decode -> RANSAC -> calibrateCamera loop. It reads
raw baseline `params.json`, scores tiny camera perturbations against independent
line_hm, and writes the same params shape for downstream evaluation.
"""
import argparse
import copy
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cached_full_test_round2 import flatten_params, smoothness
from sn_gamestate.structured_calibration.metrics import HEIGHT, WIDTH, accuracy_eval, get_polylines


DEFAULT_PARAMS = "outputs/gsr/temporal_hrnet/temporal_calib_results_hub/round2_cached_final_only/params.json"
DEFAULT_CACHE = "outputs/gsr/temporal_hrnet/round2_temporal_calib/cache_hrnet"
DEFAULT_DATA = "/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/datasets/SoccerNetGS/test"
DEFAULT_VIDEOS = [str(x) for x in list(range(116, 151)) + list(range(187, 201))]
LINE_CHANNELS = [
    "Big rect. left bottom",
    "Big rect. left main",
    "Big rect. left top",
    "Big rect. right bottom",
    "Big rect. right main",
    "Big rect. right top",
    "Goal left crossbar",
    "Goal left post left ",
    "Goal left post right",
    "Goal right crossbar",
    "Goal right post left",
    "Goal right post right",
    "Middle line",
    "Side line bottom",
    "Side line left",
    "Side line right",
    "Side line top",
    "Small rect. left bottom",
    "Small rect. left main",
    "Small rect. left top",
    "Small rect. right bottom",
    "Small rect. right main",
    "Small rect. right top",
]
LINE_TO_CHANNEL = {name.strip(): i for i, name in enumerate(LINE_CHANNELS)}


def load_baseline(path):
    data = json.load(open(path))
    return data.get("baseline", data)


def frame_from_gid(gid, vid):
    # Existing params may use official SoccerNet image ids such as 31163116000001,
    # while cached npz stores 3000000000 + video*1000000 + frame_index.
    return 3000000000 + int(vid) * 1000000 + (int(gid) % 1000000)


def build_cache_index(cache_root, vid):
    out = {}
    for p in sorted((Path(cache_root) / "test" / f"SNGS-{vid}").glob("frame_*.npz")):
        try:
            with np.load(p) as d:
                out[int(d["frame"])] = str(p)
        except Exception:
            continue
    return out


def normalize_xy(point):
    if isinstance(point, dict):
        x, y = point.get("x"), point.get("y")
    else:
        x, y = point[0], point[1]
    if x is None or y is None:
        return None
    x, y = float(x), float(y)
    if -2.0 <= x <= 2.0 and -2.0 <= y <= 2.0:
        x *= WIDTH
        y *= HEIGHT
    return x, y


def line_heatmap_score(params, line_hm, max_points=1200):
    if not isinstance(params, dict) or not params:
        return None
    try:
        polylines = get_polylines(params, WIDTH, HEIGHT, sampling_factor=0.25)
    except Exception:
        return None
    hm_ch = np.asarray(line_hm, dtype=np.float32)
    if hm_ch.shape[0] >= len(LINE_CHANNELS) + 1:
        hm_ch = hm_ch[:len(LINE_CHANNELS)]
    h, w = hm_ch.shape[1:]
    seg_scores = []
    for name, line in polylines.items():
        ch = LINE_TO_CHANNEL.get(str(name).strip())
        if ch is None or ch >= hm_ch.shape[0]:
            continue
        pts = []
        for point in line:
            xy = normalize_xy(point)
            if xy is not None:
                pts.append(xy)
        if len(pts) > max_points:
            pts = pts[:: max(1, len(pts) // max_points)]
        vals = []
        for x, y in pts:
            ix = int(round(x * (w - 1) / max(1, WIDTH - 1)))
            iy = int(round(y * (h - 1) / max(1, HEIGHT - 1)))
            if 0 <= ix < w and 0 <= iy < h:
                vals.append(float(hm_ch[ch, iy, ix]))
        if vals:
            seg_scores.append(float(np.mean(vals)))
    return float(np.mean(seg_scores)) if seg_scores else None


def candidate_params(raw, focal_scales, angle_steps, k1_candidates):
    yield "raw", copy.deepcopy(raw)
    for s in focal_scales:
        if abs(s - 1.0) < 1e-12:
            continue
        p = copy.deepcopy(raw)
        p["x_focal_length"] = float(raw["x_focal_length"]) * s
        p["y_focal_length"] = float(raw.get("y_focal_length", raw["x_focal_length"])) * s
        yield f"focal_x{s:.4f}", p
    for key, steps in angle_steps.items():
        for delta in steps:
            if abs(delta) < 1e-12:
                continue
            p = copy.deepcopy(raw)
            p[key] = float(raw[key]) + delta
            yield f"{key}_{delta:+.3f}", p
    for k1 in k1_candidates or []:
        p = copy.deepcopy(raw)
        rd = list(p.get("radial_distortion") or [0.0] * 6)
        while len(rd) < 6:
            rd.append(0.0)
        rd[0] = float(k1)
        p["radial_distortion"] = rd
        yield f"k1_{float(k1):+.3f}", p


def with_k1(params, k1):
    p = copy.deepcopy(params)
    rd = list(p.get("radial_distortion") or [0.0] * 6)
    while len(rd) < 6:
        rd.append(0.0)
    rd[0] = float(k1)
    p["radial_distortion"] = rd
    return p


def norm_delta(a, b):
    va, vb = flatten_params(a), flatten_params(b)
    if va is None or vb is None:
        return float("inf")
    scale = np.array([5, 5, 2, 500, 500, 20, 20, 10], dtype=np.float64)
    return float(np.linalg.norm((va - vb) / scale))


def choose(raw, prev, line_hm, args):
    raw_line = line_heatmap_score(raw, line_hm)
    if raw_line is None:
        return raw, raw_line, raw_line, False, "raw_line_score_failed", "raw"
    best = (raw_line, "raw", raw, raw_line, 0.0, 0.0)
    for name, cand in candidate_params(
        raw,
        focal_scales=[1.0 - args.focal_step, 1.0 + args.focal_step],
        angle_steps={
            "pan_degrees": [-args.pan_step, args.pan_step],
            "tilt_degrees": [-args.tilt_step, args.tilt_step],
            "roll_degrees": [-args.roll_step, args.roll_step],
        },
        k1_candidates=args.k1_candidates,
    ):
        line = line_heatmap_score(cand, line_hm)
        if line is None:
            continue
        trust = norm_delta(cand, raw)
        temp = norm_delta(cand, prev) if prev else 0.0
        score = line - args.lambda_trust * trust - args.lambda_temp * temp
        if score > best[0]:
            best = (score, name, cand, line, trust, temp)
    _score, name, cand, tta_line, trust, temp = best
    improved = (tta_line is not None and tta_line >= raw_line + args.margin_line)
    safe = trust <= args.max_trust_delta and (not prev or temp <= args.max_temp_delta)
    if args.gate == "none":
        use = name != "raw"
    elif args.gate == "strict":
        use = name != "raw" and improved and safe
    elif args.gate == "low_quality_only":
        raw_jump = norm_delta(raw, prev) if prev else 0.0
        low_quality = raw_line <= args.low_line_score or raw_jump >= args.low_quality_camera_delta
        use = name != "raw" and low_quality and tta_line >= raw_line - args.tolerance_line_drop and safe
    else:
        use = name != "raw" and tta_line >= raw_line - args.tolerance_line_drop and safe
    if use:
        return cand, raw_line, tta_line, True, "", name
    return raw, raw_line, tta_line, False, "gate_reject" if name != "raw" else "raw_best", name


def process_video(task):
    vid, raw_video, cfg = task
    cache_idx = build_cache_index(cfg["cache_root"], vid)
    rows, raw_params, tta_params = [], {}, {}
    prev = None
    items = sorted(raw_video.items(), key=lambda kv: int(kv[0]))
    if cfg["stride"] > 1:
        items = items[:: cfg["stride"]]
    if cfg["max_frames"]:
        items = items[: cfg["max_frames"]]
    args = argparse.Namespace(**cfg)
    video_k1 = 0.0
    if cfg.get("video_k1_candidates"):
        scores = []
        select_items = items[:: max(1, cfg.get("video_k1_select_stride", 4))]
        for k1 in [0.0] + list(cfg["video_k1_candidates"]):
            vals = []
            for gid, raw in select_items:
                frame = frame_from_gid(gid, vid)
                path = cache_idx.get(frame)
                if path is None:
                    continue
                with np.load(path) as d:
                    line_hm = d["line_hm"]
                score = line_heatmap_score(with_k1(raw, k1), line_hm)
                if score is not None:
                    vals.append(score)
            scores.append((float(np.mean(vals)) if vals else -1.0, float(k1)))
        best_score, best_k1 = max(scores)
        raw_score = next((s for s, k in scores if abs(k) < 1e-12), -1.0)
        if best_k1 and best_score >= raw_score + cfg.get("video_k1_margin", 0.001):
            video_k1 = best_k1
    for gid, raw in items:
        frame = frame_from_gid(gid, vid)
        out_gid = str(frame)
        path = cache_idx.get(frame)
        if path is None:
            rows.append({"video": vid, "frame": frame, "method": "tta_v1_fast", "solver_success": bool(raw),
                         "use_tta": False, "fallback_reason": "missing_line_hm", "raw_line_score": "",
                         "tta_line_score": "", "raw_camera_delta": "", "tta_camera_delta": "",
                         "candidate": "", "video_k1": video_k1, "focal": raw.get("x_focal_length", "") if raw else "",
                         "pan": raw.get("pan_degrees", "") if raw else "", "tilt": raw.get("tilt_degrees", "") if raw else "",
                         "roll": raw.get("roll_degrees", "") if raw else ""})
            raw_params[out_gid] = raw
            tta_params[out_gid] = raw
            continue
        with np.load(path) as d:
            line_hm = d["line_hm"]
        raw_for_tta = with_k1(raw, video_k1) if video_k1 else raw
        chosen, raw_line, tta_line, use, reason, cand_name = choose(raw_for_tta, prev, line_hm, args)
        if video_k1 and not use:
            use, reason, cand_name = True, "", f"video_k1_{video_k1:+.3f}"
        raw_params[out_gid] = raw
        tta_params[out_gid] = chosen
        rows.append({
            "video": vid,
            "frame": frame,
            "method": "tta_v1_fast",
            "solver_success": bool(chosen),
            "use_tta": use,
            "fallback_reason": reason,
            "raw_line_score": raw_line if raw_line is not None else "",
            "tta_line_score": tta_line if tta_line is not None else "",
            "raw_camera_delta": norm_delta(raw, prev) if prev else "",
            "tta_camera_delta": norm_delta(chosen, prev) if prev else "",
            "candidate": cand_name,
            "video_k1": video_k1,
            "focal": chosen.get("x_focal_length", "") if chosen else "",
            "pan": chosen.get("pan_degrees", "") if chosen else "",
            "tilt": chosen.get("tilt_degrees", "") if chosen else "",
            "roll": chosen.get("roll_degrees", "") if chosen else "",
        })
        prev = chosen
    raw_sm, tta_sm = smoothness(raw_params), smoothness(tta_params)
    vr = {
        "video": vid,
        "frames": len(rows),
        "use_tta_ratio": sum(bool(r["use_tta"]) for r in rows) / max(1, len(rows)),
        "raw_line_score_mean": float(np.mean([float(r["raw_line_score"]) for r in rows if r["raw_line_score"] != ""])) if rows else None,
        "tta_line_score_mean": float(np.mean([float(r["tta_line_score"]) for r in rows if r["tta_line_score"] != ""])) if rows else None,
        "raw_smooth_mean": raw_sm["mean"],
        "tta_smooth_mean": tta_sm["mean"],
        "raw_smooth_p95": raw_sm["p95"],
        "tta_smooth_p95": tta_sm["p95"],
        "video_k1": video_k1,
    }
    return vid, rows, raw_params, tta_params, vr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-params", default=DEFAULT_PARAMS)
    ap.add_argument("--cache-root", default=DEFAULT_CACHE)
    ap.add_argument("--data-root", default=DEFAULT_DATA)
    ap.add_argument("--out", default="outputs/tta_calib/tta_v1_camera/fast_probe")
    ap.add_argument("--video-list", nargs="+", default=None)
    ap.add_argument("--max-videos", type=int, default=-1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=20, help="Subsample existing params by order, not raw frame ids.")
    ap.add_argument("--gate", choices=["none", "safe", "strict", "low_quality_only"], default="safe")
    ap.add_argument("--focal-step", type=float, default=0.01)
    ap.add_argument("--pan-step", type=float, default=0.05)
    ap.add_argument("--tilt-step", type=float, default=0.03)
    ap.add_argument("--roll-step", type=float, default=0.03)
    ap.add_argument("--k1-candidates", nargs="*", type=float, default=[])
    ap.add_argument("--video-k1-candidates", nargs="*", type=float, default=[])
    ap.add_argument("--video-k1-margin", type=float, default=0.001)
    ap.add_argument("--video-k1-select-stride", type=int, default=4)
    ap.add_argument("--lambda-trust", type=float, default=0.02)
    ap.add_argument("--lambda-temp", type=float, default=0.01)
    ap.add_argument("--margin-line", type=float, default=0.002)
    ap.add_argument("--tolerance-line-drop", type=float, default=0.0005)
    ap.add_argument("--max-trust-delta", type=float, default=2.0)
    ap.add_argument("--max-temp-delta", type=float, default=4.0)
    ap.add_argument("--low-line-score", type=float, default=0.027)
    ap.add_argument("--low-quality-camera-delta", type=float, default=3.0)
    ap.add_argument("--official-eval", action="store_true")
    ap.add_argument("--nproc", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    raw_by_video = load_baseline(args.input_params)
    videos = args.video_list or [v for v in DEFAULT_VIDEOS if v in raw_by_video]
    if args.max_videos and args.max_videos > 0:
        videos = videos[: args.max_videos]

    cfg = vars(args).copy()
    tasks = [(vid, raw_by_video.get(vid, {}), cfg) for vid in videos]
    if args.nproc and args.nproc > 1 and len(tasks) > 1:
        with Pool(min(args.nproc, len(tasks))) as pool:
            parts = pool.map(process_video, tasks)
    else:
        parts = [process_video(t) for t in tasks]

    rows, raw_params, tta_params, video_rows = [], {}, {}, []
    for vid, vrows, vraw, vtta, vmetrics in parts:
        rows.extend(vrows)
        raw_params[vid] = vraw
        tta_params[vid] = vtta
        video_rows.append(vmetrics)
        (pred_dir / f"SNGS-{vid}.json").write_text(json.dumps(vtta, indent=2), encoding="utf-8")

    fields = list(rows[0]) if rows else ["video"]
    with (out / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with (out / "video_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(video_rows[0]) if video_rows else ["video"])
        w.writeheader()
        w.writerows(video_rows)

    params_out = {"baseline_raw": raw_params, "tta_v1_fast": tta_params}
    (out / "params.json").write_text(json.dumps(params_out), encoding="utf-8")

    result = {"videos": videos, "n_frames": len(rows), "video_metrics": video_rows, "official_eval": None}
    if args.official_eval:
        result["official_eval"] = {
            "baseline_raw": accuracy_eval(raw_params, args.data_root, videos, nproc=args.nproc, stride=1),
            "tta_v1_fast": accuracy_eval(tta_params, args.data_root, videos, nproc=args.nproc, stride=1),
        }
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "config.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Fast Camera-Level TTA Probe",
        "",
        f"- input_params: {args.input_params}",
        "- line_score: semantic per-channel NBJW line_hm score; circle primitives skipped",
        f"- frames: {len(rows)}",
        f"- gate: {args.gate}",
        f"- official_eval: {bool(args.official_eval)}",
        "",
        "| video | frames | use_tta | raw_line | tta_line | raw_smooth | tta_smooth |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in video_rows:
        def fmt(x):
            return "" if x is None else f"{float(x):.5f}"
        lines.append(
            f"| {r['video']} | {r['frames']} | {r['use_tta_ratio']:.3f} | {fmt(r['raw_line_score_mean'])} | "
            f"{fmt(r['tta_line_score_mean'])} | {fmt(r['raw_smooth_mean'])} | {fmt(r['tta_smooth_mean'])} |"
        )
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((out / "RESULTS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
