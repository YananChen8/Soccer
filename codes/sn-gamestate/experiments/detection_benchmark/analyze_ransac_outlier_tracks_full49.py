#!/usr/bin/env python3
"""Frame-level RANSAC outlier track diagnostics on the cached full49 test set.

K is measured in evaluated stride steps. With the default stride=20, K=3 means
looking up to 60 raw frames before/after the current evaluated frame.
"""
import argparse
import csv
import json
import glob
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from cached_full_test_round2 import (
    KeypointKalman,
    _score_job,
    complete_keypoints,
    coords_to_dict,
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    ransac_inlier_keys,
    solve_params,
    smoothness,
)


VIDEOS = [str(x) for x in list(range(116, 151)) + list(range(187, 201))]


def decode_npz(path):
    d = np.load(path)
    kp = torch.from_numpy(d["kp_hm"].astype(np.float32)).unsqueeze(0)
    line = torch.from_numpy(d["line_hm"].astype(np.float32)).unsqueeze(0)
    kc = get_keypoints_from_heatmap_batch_maxpool(kp[:, :-1])
    lc = get_keypoints_from_heatmap_batch_maxpool_l(line[:, :-1])
    pred = complete_keypoints(
        coords_to_dict(kc, threshold=0.1449),
        coords_to_dict(lc, threshold=0.2983),
        w=960,
        h=540,
        normalize=True,
    )[0]
    return int(d["frame"]), pred


def load_video(cache_root, vid, stride):
    files = sorted(glob.glob(str(Path(cache_root) / "test" / f"SNGS-{vid}" / "frame_*.npz")))
    rows = []
    for i, f in enumerate(files):
        if i % stride != 0:
            continue
        frame, kps = decode_npz(f)
        inliers = ransac_inlier_keys(kps)
        rows.append({
            "video": vid,
            "idx": len(rows),
            "frame": frame,
            "image_id": f"3{vid}{frame % 1000000:06d}",
            "image_file": f"{frame % 1000000:06d}.jpg",
            "kps": kps,
            "inliers": set(inliers),
            "outliers": set(kps) - set(inliers),
        })
    return rows


def context_for(rows, idx, key, k):
    prev, nxt = [], []
    for off in range(1, k + 1):
        j = idx - off
        if j >= 0 and key in rows[j]["kps"]:
            prev.append((off, rows[j]["frame"], rows[j]["kps"][key]))
        j = idx + off
        if j < len(rows) and key in rows[j]["kps"]:
            nxt.append((off, rows[j]["frame"], rows[j]["kps"][key]))
    return prev, nxt


def replacement_from_context(rows, idx, key, k):
    prev, nxt = context_for(rows, idx, key, k)
    candidates = []
    for off, _frame, pt in prev + nxt:
        candidates.append((off, pt))
    if not candidates:
        return None
    off, pt = sorted(candidates, key=lambda x: x[0])[0]
    return {"x": float(pt["x"]), "y": float(pt["y"]), "p": float(pt.get("p", 1.0))}


def solve_variants_for_video(args):
    cache_root, vid, stride, ks = args
    rows = load_video(cache_root, vid, stride)
    params = {"baseline": {}, "kalman_original": {}}
    params.update({f"oracle_copy_k{k}": {} for k in ks})
    stats = {name: defaultdict(int) for name in params}
    kf = KeypointKalman()
    for row in rows:
        base = row["kps"]
        outliers = set(row["outliers"])
        params["baseline"][row["image_id"]] = solve_params(base)
        stats["baseline"]["detected_points"] += len(base)
        stats["baseline"]["initial_outliers"] += len(outliers)

        kal = {k: dict(v) for k, v in base.items()}
        repl = 0
        for key in outliers:
            p = kf.predict(key)
            if p is not None:
                kal[key]["x"], kal[key]["y"] = p
                repl += 1
        still = set(kal) - ransac_inlier_keys(kal)
        for key in still:
            kal.pop(key, None)
        kf.update_many(kal)
        params["kalman_original"][row["image_id"]] = solve_params(kal)
        stats["kalman_original"]["detected_points"] += len(base)
        stats["kalman_original"]["initial_outliers"] += len(outliers)
        stats["kalman_original"]["replaced"] += repl
        stats["kalman_original"]["removed"] += len(still)

        for k in ks:
            name = f"oracle_copy_k{k}"
            cur = {kk: dict(vv) for kk, vv in base.items()}
            repl = 0
            for key in outliers:
                p = replacement_from_context(rows, row["idx"], key, k)
                if p is not None:
                    cur[key].update(p)
                    repl += 1
            still = set(cur) - ransac_inlier_keys(cur)
            for key in still:
                cur.pop(key, None)
            params[name][row["image_id"]] = solve_params(cur)
            stats[name]["detected_points"] += len(base)
            stats[name]["initial_outliers"] += len(outliers)
            stats[name]["replaced"] += repl
            stats[name]["removed"] += len(still)
    sm = {name: smoothness(pb) for name, pb in params.items()}
    return vid, params, {k: dict(v) for k, v in stats.items()}, sm, rows


def pack_point(pt):
    return {"x": float(pt["x"]), "y": float(pt["y"]), "p": float(pt.get("p", 1.0))}


def build_events_and_cases(all_rows_by_video, ks, case_count):
    max_k = max(ks)
    events = []
    for vid, rows in all_rows_by_video.items():
        for row in rows:
            for key in sorted(row["outliers"]):
                prev, nxt = context_for(rows, row["idx"], key, max_k)
                ev = {
                    "video": vid,
                    "frame": row["frame"],
                    "image_id": row["image_id"],
                    "image_file": row["image_file"],
                    "idx": row["idx"],
                    "key": key,
                    "x": float(row["kps"][key]["x"]),
                    "y": float(row["kps"][key]["y"]),
                    "p": float(row["kps"][key].get("p", 1.0)),
                    "prev_count_max_k": len(prev),
                    "next_count_max_k": len(nxt),
                    "recoverable_max_k": int(bool(prev or nxt)),
                }
                for k in ks:
                    p, n = context_for(rows, row["idx"], key, k)
                    ev[f"prev_within_k{k}"] = int(bool(p))
                    ev[f"next_within_k{k}"] = int(bool(n))
                    ev[f"either_within_k{k}"] = int(bool(p or n))
                events.append(ev)
    frame_rank = defaultdict(int)
    for ev in events:
        frame_rank[(ev["video"], ev["idx"])] += 1
    chosen = sorted(frame_rank.items(), key=lambda kv: kv[1], reverse=True)[:case_count]
    cases = []
    for (vid, idx), nout in chosen:
        row = all_rows_by_video[vid][idx]
        outlier_payload = []
        for key in sorted(row["outliers"]):
            prev, nxt = context_for(all_rows_by_video[vid], idx, key, max_k)
            outlier_payload.append({
                "key": key,
                "point": pack_point(row["kps"][key]),
                "prev": [{"offset": off, "frame": fr, "point": pack_point(pt)} for off, fr, pt in prev],
                "next": [{"offset": off, "frame": fr, "point": pack_point(pt)} for off, fr, pt in nxt],
            })
        cases.append({
            "video": vid,
            "frame": row["frame"],
            "image_id": row["image_id"],
            "image_file": row["image_file"],
            "n_outliers": nout,
            "inliers": {str(k): pack_point(row["kps"][k]) for k in row["inliers"] if k in row["kps"]},
            "outliers": outlier_payload,
            "image_path": f"datasets/SoccerNetGS/test/SNGS-{vid}/img1/{row['image_file']}",
        })
    return events, cases


def aggregate_scores(params, nproc):
    jobs = [(vid, name, params[name][vid]) for name in params for vid in params[name]]
    with Pool(nproc) as pool:
        parts = pool.map(_score_job, jobs)
    agg = {name: {"micro": [], "macro": [], "reproj": [], "nvid": 0} for name in params}
    for name, _vid, mi, ma, rj in parts:
        agg[name]["micro"] += mi
        agg[name]["macro"] += ma
        agg[name]["reproj"] += rj
        agg[name]["nvid"] += 1
    return {
        name: {
            "point_acc": float(np.mean(v["micro"])) if v["micro"] else None,
            "line_acc": float(np.mean(v["macro"])) if v["macro"] else None,
            "reproj_mean": float(np.mean(v["reproj"])) if v["reproj"] else None,
            "n_frames": len(v["micro"]),
            "n_videos": v["nvid"],
        }
        for name, v in agg.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default="outputs/gsr/temporal_hrnet/round2_temporal_calib/cache_hrnet")
    ap.add_argument("--out-dir", default="outputs/gsr/temporal_hrnet/temporal_calib_results_hub/failure_analysis/ransac_outlier_tracks_full49")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10, 20, 40])
    ap.add_argument("--videos", nargs="+", default=VIDEOS)
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--case-count", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(args.cache_root, vid, args.stride, args.ks) for vid in args.videos]
    params = {}
    stats = defaultdict(lambda: defaultdict(int))
    sm = defaultdict(list)
    all_rows_by_video = {}
    with Pool(min(args.nproc, len(jobs))) as pool:
        for vid, p, st, s, rows in pool.imap_unordered(solve_variants_for_video, jobs):
            print(f"video={vid} frames={len(rows)}", flush=True)
            all_rows_by_video[vid] = rows
            for name, pb in p.items():
                params.setdefault(name, {})[vid] = pb
            for name, vals in st.items():
                for k, v in vals.items():
                    stats[name][k] += v
            for name, val in s.items():
                sm[name].append(val)

    events, cases = build_events_and_cases(all_rows_by_video, args.ks, args.case_count)
    with open(out / "outlier_events.csv", "w", newline="") as f:
        fieldnames = list(events[0])
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(events)
    json.dump(cases, open(out / "case_data.json", "w"), indent=2)

    k_rows = []
    total_out = len(events)
    for k in args.ks:
        either = sum(ev[f"either_within_k{k}"] for ev in events)
        prev = sum(ev[f"prev_within_k{k}"] for ev in events)
        nxt = sum(ev[f"next_within_k{k}"] for ev in events)
        k_rows.append({
            "k_stride_steps": k,
            "raw_frame_window": k * args.stride,
            "outliers": total_out,
            "prev_recoverable": prev,
            "next_recoverable": nxt,
            "either_recoverable": either,
            "not_recoverable": total_out - either,
            "either_ratio": either / total_out if total_out else 0.0,
        })
    with open(out / "k_recoverability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(k_rows[0]))
        w.writeheader()
        w.writerows(k_rows)

    score = aggregate_scores(params, args.nproc)
    result = {}
    for name, sc in score.items():
        mean_sm = [x["mean"] for x in sm[name] if x["mean"] is not None]
        p95_sm = [x["p95"] for x in sm[name] if x["p95"] is not None]
        sc["smoothness_mean"] = float(np.mean(mean_sm)) if mean_sm else None
        sc["smoothness_p95"] = float(np.mean(p95_sm)) if p95_sm else None
        sc["mode_stats"] = dict(stats[name])
        result[name] = sc
    json.dump(result, open(out / "metric_k_sweep_oracle_copy.json", "w"), indent=2)

    md = [
        "# RANSAC Outlier Track Diagnostics Full49",
        "",
        f"videos={len(args.videos)}, stride={args.stride}, scored_frames={sum(len(v) for v in all_rows_by_video.values())}",
        f"outlier_events={total_out}",
        "",
        "K is measured in evaluated stride steps. `oracle_copy_k*` is non-causal and copies nearest same-id keypoint from neighbor frames; it is a diagnostic upper-bound-style test, not deployable Kalman.",
        "",
        "## Recoverability",
        "",
        "| K | raw-frame window | either recoverable | not recoverable | ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in k_rows:
        md.append(f"| {r['k_stride_steps']} | {r['raw_frame_window']} | {r['either_recoverable']} | {r['not_recoverable']} | {r['either_ratio']:.3f} |")
    md += ["", "## Metrics", "", "| method | point | line | reproj | replaced | removed |", "|---|---:|---:|---:|---:|---:|"]
    for name in sorted(result):
        r = result[name]
        st = r["mode_stats"]
        md.append(f"| {name} | {r['point_acc']:.4f} | {r['line_acc']:.4f} | {r['reproj_mean']:.2f} | {st.get('replaced', 0)} | {st.get('removed', 0)} |")
    (out / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print((out / "README.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
