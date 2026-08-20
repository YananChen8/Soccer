"""Tiny dense-flow camera-prior test.

Reads existing BroadTrack-style params and asks one question:
does dense optical flow of the previous field projection pick a better current
camera candidate than raw NBJW? This is a cheap proxy for "flow as camera init".
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from broadtrack_min_ablation_round3 import image_path_for_frame
from cached_full_test_round2 import _score_job, smoothness


def gid_frame(gid):
    return int(gid[-6:])


def load_gray(vid, gid, size):
    img = cv2.imread(image_path_for_frame(vid, gid_frame(gid)), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def line_points(params, width, height):
    if not params:
        return {}
    from sn_gamestate.structured_calibration.metrics import get_polylines

    try:
        polys = get_polylines(params, width, height, sampling_factor=0.9)
    except Exception:
        return {}
    out = {}
    for name, pts in polys.items():
        if pts and isinstance(pts[0], dict):
            arr = np.asarray([[p["x"], p["y"]] for p in pts], dtype=np.float32)
        else:
            arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        arr = arr[(arr[:, 0] >= -width) & (arr[:, 0] <= 2 * width) & (arr[:, 1] >= -height) & (arr[:, 1] <= 2 * height)]
        if len(arr) >= 2:
            out[name] = arr
    return out


def sample_rows(arr, max_points=80):
    if len(arr) <= max_points:
        return arr
    idx = np.linspace(0, len(arr) - 1, max_points).astype(np.int32)
    return arr[idx]


def propagate_lines(prev_lines, flow, width, height):
    propagated = {}
    for name, pts in prev_lines.items():
        pts = sample_rows(pts)
        xs = np.clip(np.rint(pts[:, 0]).astype(np.int32), 0, width - 1)
        ys = np.clip(np.rint(pts[:, 1]).astype(np.int32), 0, height - 1)
        moved = pts + flow[ys, xs]
        keep = (moved[:, 0] >= -width) & (moved[:, 0] <= 2 * width) & (moved[:, 1] >= -height) & (moved[:, 1] <= 2 * height)
        if int(keep.sum()) >= 2:
            propagated[name] = moved[keep]
    return propagated


def nearest_mean(a, b):
    if len(a) == 0 or len(b) == 0:
        return None
    # ponytail: quadratic is fine for <=80 sampled points/line; replace with KD-tree if this grows.
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    return float(np.mean(np.min(d, axis=1)))


def flow_line_error(target_lines, cand_lines):
    vals = []
    for name, a in target_lines.items():
        b = cand_lines.get(name)
        if b is None:
            continue
        v = nearest_mean(a, sample_rows(b))
        if v is not None:
            vals.append(v)
    return None if not vals else float(np.mean(vals))


def select_video(vid, params, methods, width, height, gate_px, allow_prev_init=False):
    gids = sorted(params[methods[0]].get(vid, {}))
    selected, prev_gid, prev_params, stats = {}, None, None, Counter()
    rows = []
    for gid in gids:
        base = params.get("baseline", {}).get(vid, {}).get(gid, {})
        if not base:
            selected[gid] = {}
            continue
        candidates = {"baseline": base}
        for m in methods:
            p = params.get(m, {}).get(vid, {}).get(gid, {})
            if p:
                candidates[m] = p
        if allow_prev_init and prev_gid and prev_params:
            candidates["prev_init"] = prev_params

        choice, errors = "baseline", {}
        if prev_gid and prev_params:
            prev_gray = load_gray(vid, prev_gid, (width, height))
            cur_gray = load_gray(vid, gid, (width, height))
            prev_lines = line_points(prev_params, width, height)
            if prev_gray is not None and cur_gray is not None and prev_lines:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 25, 3, 5, 1.2, 0)
                target = propagate_lines(prev_lines, flow, width, height)
                if target:
                    for name, cand in candidates.items():
                        errors[name] = flow_line_error(target, line_points(cand, width, height))
                    valid = {k: v for k, v in errors.items() if v is not None}
                    if valid:
                        best = min(valid, key=valid.get)
                        # Keep a simple trust region: only switch if the flow fit wins by gate_px.
                        if best != "baseline" and valid[best] + gate_px < valid.get("baseline", 1e9):
                            choice = best
        selected[gid] = candidates[choice]
        prev_gid, prev_params = gid, selected[gid] or base
        stats[f"chosen_{choice}"] += 1
        if errors:
            rows.append({"video": vid, "gid": gid, "choice": choice, **{f"flowerr_{k}": v for k, v in errors.items()}})
    return selected, dict(stats), rows


def aggregate(parts):
    out = {"micro": [], "macro": [], "reproj": [], "nvid": 0}
    for _method, _vid, mi, ma, rj in parts:
        out["micro"] += mi
        out["macro"] += ma
        out["reproj"] += rj
        out["nvid"] += 1
    return {
        "point_acc": float(np.mean(out["micro"])) if out["micro"] else None,
        "line_acc": float(np.mean(out["macro"])) if out["macro"] else None,
        "reproj_mean": float(np.mean(out["reproj"])) if out["reproj"] else None,
        "n_frames": len(out["micro"]),
        "n_videos": out["nvid"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--videos", nargs="+", default=["116", "117", "118"])
    ap.add_argument("--methods", nargs="+", default=["radial_k1"])
    ap.add_argument("--gate-px", type=float, default=2.0)
    ap.add_argument("--allow-prev-init", action="store_true")
    args = ap.parse_args()

    from sn_gamestate.structured_calibration.metrics import HEIGHT, WIDTH

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = json.load(open(args.params_json))
    selected, stats, rows = {}, Counter(), []
    for vid in args.videos:
        pb, st, rr = select_video(vid, params, ["baseline"] + args.methods, WIDTH, HEIGHT, args.gate_px, args.allow_prev_init)
        selected[vid] = pb
        stats.update(st)
        rows.extend(rr)

    payload = {"dense_flow_select": selected}
    json.dump(payload, open(out_dir / "params_dense_flow_select.json", "w"))
    parts = [_score_job((vid, "dense_flow_select", selected[vid])) for vid in args.videos]
    res = aggregate(parts)
    res["smoothness"] = {vid: smoothness(selected[vid]) for vid in args.videos}
    res["stats"] = dict(stats)
    json.dump(res, open(out_dir / "result_dense_flow_select.json", "w"), indent=2)

    # Score original baselines on the same subset for direct comparison.
    cmp = {"dense_flow_select": res}
    for method in ["baseline"] + args.methods:
        parts = [_score_job((vid, method, params[method][vid])) for vid in args.videos if vid in params.get(method, {})]
        cmp[method] = aggregate(parts)
        cmp[method]["smoothness"] = {vid: smoothness(params[method][vid]) for vid in args.videos if vid in params.get(method, {})}

    import csv

    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(out_dir / "frame_flow_errors.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    json.dump(cmp, open(out_dir / "comparison.json", "w"), indent=2)

    def fmt(x, n=4):
        return "NA" if x is None else f"{x:.{n}f}"

    lines = [
        "# Dense Flow Init Probe",
        "",
        f"- videos: {', '.join(args.videos)}",
        f"- gate_px: {args.gate_px}",
        f"- selection stats: `{json.dumps(dict(stats), sort_keys=True)}`",
        "",
        "| method | point | line | reproj | frames |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ["baseline"] + args.methods + ["dense_flow_select"]:
        r = cmp.get(method, {})
        lines.append(f"| {method} | {fmt(r.get('point_acc'))} | {fmt(r.get('line_acc'))} | {fmt(r.get('reproj_mean'), 2)} | {r.get('n_frames')} |")
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((out_dir / "RESULTS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
