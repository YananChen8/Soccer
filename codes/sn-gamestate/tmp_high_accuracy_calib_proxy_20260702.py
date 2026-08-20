import argparse
import csv
import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import tmp_official_aux_report_eval_visual_20260701 as ref
from experiments.detection_benchmark import eval_temporal_input_mixer_calib as base
from nbjw_calib.utils.utils_linesWC import LineKeypointsWCDB
from sn_gamestate.structured_calibration.metrics import HEIGHT, WIDTH, get_polylines


FEATURE_W = 64
FEATURE_H = 36
LINE_NAMES = [n.strip() for n in LineKeypointsWCDB(Image.new("RGB", (960, 540)), np.eye(3), (960, 540)).lines_list]
LINE_TO_CH = {n: i for i, n in enumerate(LINE_NAMES)}


def add_gaussian(mask, x, y, sigma=1.5):
    h, w = mask.shape
    if not np.isfinite(x) or not np.isfinite(y):
        return
    if x < 0 or x >= w or y < 0 or y >= h:
        return
    r = max(2, int(np.ceil(3 * sigma)))
    x0, x1 = max(0, int(round(x)) - r), min(w, int(round(x)) + r + 1)
    y0, y1 = max(0, int(round(y)) - r), min(h, int(round(y)) + r + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    patch = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma)).astype(np.float32)
    mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], patch)


def normalize_feature(x):
    x = x.astype(np.float32).reshape(-1)
    x -= float(x.mean())
    n = float(np.linalg.norm(x))
    return x / n if n > 1e-8 else x


def render_params_feature(params, width=FEATURE_W, height=FEATURE_H):
    mask = np.zeros((len(LINE_NAMES), height, width), np.float32)
    if not params:
        return normalize_feature(mask)
    try:
        lines = get_polylines(params, WIDTH, HEIGHT, sampling_factor=0.9)
    except Exception:
        return normalize_feature(mask)
    sx = width / float(WIDTH)
    sy = height / float(HEIGHT)
    for name, pts in lines.items():
        ch = LINE_TO_CH.get(str(name).strip())
        if ch is None:
            continue
        if pts and isinstance(pts[0], dict):
            arr = np.asarray([[p["x"], p["y"]] for p in pts], dtype=np.float32)
        else:
            arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        pix = arr * np.array([sx, sy], dtype=np.float32)
        add_gaussian(mask[ch], pix[0, 0], pix[0, 1])
        add_gaussian(mask[ch], pix[-1, 0], pix[-1, 1])
    return normalize_feature(mask)


def query_feature_from_line_hm(line_hm):
    # NBJW line heatmap is the segmentation proxy: semantic line evidence, no cGAN training.
    x = line_hm[:, :len(LINE_NAMES)].detach().float()
    x = torch.nn.functional.interpolate(x, size=(FEATURE_H, FEATURE_W), mode="bilinear", align_corners=False)
    arr = x.squeeze(0).cpu().numpy()
    for ch in range(arr.shape[0]):
        arr[ch] -= float(arr[ch].min())
        mx = float(arr[ch].max())
        if mx > 1e-8:
            arr[ch] /= mx
        arr[ch] = arr[ch] ** 2
    return normalize_feature(arr)


def average_params(rows):
    vals = {}
    for row in rows:
        params = row.get("params") or {}
        for k, v in params.items():
            if isinstance(v, (int, float)) and np.isfinite(float(v)):
                vals.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in vals.items() if v}


def sample_candidate_params(anchor_rows, db_size, seed):
    rng = np.random.default_rng(seed)
    anchors = [r.get("params") for r in anchor_rows if r.get("params")]
    if not anchors:
        return []
    keys = [k for k in ["pan_degrees", "tilt_degrees", "roll_degrees", "x_focal_length", "y_focal_length"] if any(k in p for p in anchors)]
    arr = np.array([[float(p.get(k, np.nan)) for k in keys] for p in anchors], dtype=np.float32)
    center = np.nanmean(arr, axis=0)
    spread = np.nanstd(arr, axis=0)
    spread = np.where(np.isfinite(spread) & (spread > 1e-5), spread, np.maximum(np.abs(center) * 0.03, 1.0))
    out = []
    # Keep the exact solved cameras as valid database anchors; sampled variants inherit
    # non-scalar fields such as principal_point and position_meters.
    out.extend(copy.deepcopy(anchors[: min(len(anchors), db_size)]))
    for _ in range(max(0, db_size - 1)):
        j = int(rng.integers(0, len(anchors)))
        cand = copy.deepcopy(anchors[j])
        base_vec = np.array([float(cand.get(k, center[i])) for i, k in enumerate(keys)], dtype=np.float32)
        vec = base_vec + rng.normal(0.0, spread * 0.75)
        for k, v in zip(keys, vec):
            if np.isfinite(v):
                cand[k] = float(v)
        if "x_focal_length" in cand:
            cand["y_focal_length"] = float(cand["x_focal_length"])
        out.append(cand)
    return out[:db_size]


def camera_smooth_l2(rows):
    vecs = []
    keys = sorted({k for r in rows for k, v in (r.get("params") or {}).items() if isinstance(v, (int, float))})
    for r in rows:
        p = r.get("params") or {}
        if p:
            vecs.append(np.array([float(p.get(k, 0.0)) for k in keys], dtype=np.float32))
    jumps = [float(np.linalg.norm(b - a)) for a, b in zip(vecs, vecs[1:])]
    return (float(np.mean(jumps)), float(np.percentile(jumps, 95))) if jumps else (None, None)


def summarize(rows):
    reproj = [x for r in rows for x in (r.get("reproj") or [])]
    scored = [r for r in rows if r.get("reproj_mean") is not None]
    smooth_mean, smooth_p95 = camera_smooth_l2(rows)
    jac5 = ref.mean([1.0 if x <= 5.0 else 0.0 for x in reproj])
    cr = len(scored) / len(rows) if rows else None
    return {
        "point_acc": ref.mean([r.get("point_acc") for r in rows]),
        "line_acc": ref.mean([r.get("line_acc") for r in rows]),
        "reproj_mean": ref.mean([r.get("reproj_mean") for r in rows]),
        "JaC@5": jac5,
        "JaC@10": ref.mean([1.0 if x <= 10.0 else 0.0 for x in reproj]),
        "JaC@15": ref.mean([1.0 if x <= 15.0 else 0.0 for x in reproj]),
        "JaC@20": ref.mean([1.0 if x <= 20.0 else 0.0 for x in reproj]),
        "MRE": ref.mean(reproj),
        "CR": cr,
        "Final Score": cr * jac5 if cr is not None and jac5 is not None else None,
        "camera_smooth_l2_mean": smooth_mean,
        "camera_smooth_l2_p95": smooth_p95,
        "n_total": len(rows),
        "n_scored": len(scored),
    }


def score_params(params, gt_lines):
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


def write_tables(results, out_dir):
    keys = ["point_acc", "line_acc", "reproj_mean", "JaC@5", "JaC@10", "JaC@15", "JaC@20", "MRE", "CR", "Final Score", "camera_smooth_l2_mean", "camera_smooth_l2_p95", "n_total", "n_scored"]
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run"] + keys)
        for run, data in results.items():
            row = data["aggregate"]
            w.writerow([run] + [row.get(k) for k in keys])


def evaluate(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    frames_root, data_root = ref.get_split_paths("test")
    videos = [str(v).replace("SNGS-", "") for v in args.videos]

    print(f"loading_hrnets device={device}", flush=True)
    kp_model, line_model = base.load_hrnets(device)
    print("loaded_hrnets", flush=True)
    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    raw_rows = []
    query_rows = []
    frame_cache = []
    t0 = time.perf_counter()

    for video in videos:
        files = sorted((frames_root / f"SNGS-{video}" / "img1").glob("*.jpg"))
        gt = base.load_gt_lines_for_video(str(data_root), video)
        id_map = ref.image_id_map(data_root, video)
        for idx, image_path in enumerate(files):
            if idx % args.stride != 0:
                continue
            gid = id_map.get(image_path.stem, f"3{video}{image_path.stem}")
            if gid not in gt:
                continue
            if args.max_samples and len(query_rows) >= args.max_samples:
                break
            img = tfm(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                kp_hm = kp_model(img)
                line_hm = line_model(img)
            raw_score = ref.score_hm(kp_hm, line_hm, gt[gid])
            raw_row = {"run": "raw_nbjw", "video": video, "frame": image_path.stem, **raw_score}
            raw_rows.append(raw_row)
            query_rows.append({
                "video": video,
                "frame": image_path.stem,
                "query_feature": query_feature_from_line_hm(line_hm),
                "gt": gt[gid],
                "anchor_index": len(frame_cache),
            })
            frame_cache.append(raw_row)
        print(f"collected video={video} raw_rows={len(raw_rows)}", flush=True)
        if args.max_samples and len(query_rows) >= args.max_samples:
            break

    feature_dim = len(LINE_NAMES) * FEATURE_W * FEATURE_H
    print(f"building_db candidates={args.db_size} anchors={len(frame_cache)} feature_source={args.db_feature_source}", flush=True)
    if args.db_feature_source == "anchor_heatmap":
        candidates = [copy.deepcopy(r.get("params") or {}) for r in frame_cache]
        cand_features = np.stack([r["query_feature"] for r in query_rows]).astype(np.float32) if query_rows else np.empty((0, feature_dim), np.float32)
    else:
        candidates = sample_candidate_params(frame_cache, args.db_size, args.seed)
        cand_features = np.stack([render_params_feature(p) for p in candidates]).astype(np.float32) if candidates else np.empty((0, feature_dim), np.float32)
    db_build_seconds = time.perf_counter() - t0
    retrieval_rows = []
    rt0 = time.perf_counter()
    print(f"retrieving frames={len(query_rows)} db={len(candidates)}", flush=True)
    for row in query_rows:
        q = row["query_feature"]
        d = np.linalg.norm(cand_features - q[None, :], axis=1) if len(cand_features) else np.array([])
        if args.leave_one_out and args.db_feature_source == "anchor_heatmap" and int(row["anchor_index"]) < len(d):
            d[int(row["anchor_index"])] = np.inf
        best = int(np.argmin(d)) if len(d) else -1
        oracle_idx = int(row["anchor_index"]) if int(row["anchor_index"]) < len(d) else -1
        oracle_rank = None
        oracle_distance = None
        if oracle_idx >= 0:
            oracle_distance = float(d[oracle_idx])
            oracle_rank = int(np.sum(d < d[oracle_idx]) + 1)
        params = candidates[best] if best >= 0 else {}
        score = score_params(params, row["gt"])
        retrieval_rows.append({
            "run": "paper_proxy_nn",
            "video": row["video"],
            "frame": row["frame"],
            "nn_index": best,
            "nn_distance": float(d[best]) if best >= 0 else None,
            "oracle_index": oracle_idx,
            "oracle_rank": oracle_rank,
            "oracle_distance": oracle_distance,
            **score,
        })
    retrieval_seconds = time.perf_counter() - rt0

    results = {
        "raw_nbjw": {"aggregate": summarize(raw_rows)},
        "paper_proxy_nn": {"aggregate": summarize(retrieval_rows)},
    }
    meta = {
        "db_size": len(candidates),
        "feature": f"23-channel semantic NBJW line feature, source={args.db_feature_source}, {FEATURE_W}x{FEATURE_H}, L2-normalized",
        "db_feature_source": args.db_feature_source,
        "leave_one_out": bool(args.leave_one_out),
        "retrieval": "brute-force L2 nearest neighbor",
        "refinement": "disabled",
        "oracle_top1": ref.mean([1.0 if r.get("oracle_rank") == 1 else 0.0 for r in retrieval_rows]),
        "oracle_median_rank": float(np.median([r["oracle_rank"] for r in retrieval_rows if r.get("oracle_rank") is not None])) if retrieval_rows else None,
        "device": str(device),
        "videos": videos,
        "stride": args.stride,
        "db_build_seconds": db_build_seconds,
        "retrieval_seconds": retrieval_seconds,
        "retrieval_ms_per_frame": 1000.0 * retrieval_seconds / max(1, len(query_rows)),
    }
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    write_tables(results, out_dir)
    with (out_dir / "frame_scores.csv").open("w", newline="") as f:
        fields = ["run", "video", "frame", "point_acc", "line_acc", "reproj_mean", "nn_index", "nn_distance", "oracle_index", "oracle_rank", "oracle_distance"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(raw_rows)
        w.writerows(retrieval_rows)
    (out_dir / "db_candidates.json").write_text(json.dumps(candidates[: min(len(candidates), 200)], indent=2))
    print(json.dumps({"meta": meta, "summary": {k: v["aggregate"] for k, v in results.items()}}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["116"])
    ap.add_argument("--stride", type=int, default=80)
    ap.add_argument("--db-size", type=int, default=1000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--db-feature-source", choices=["rendered_params", "anchor_heatmap"], default="rendered_params")
    ap.add_argument("--leave-one-out", action="store_true")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
