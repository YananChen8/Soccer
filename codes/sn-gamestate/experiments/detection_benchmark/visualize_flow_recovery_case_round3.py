"""Visualize the frame where flow repairs the most RANSAC-outlier keypoints."""
import argparse
import csv
import json
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from broadtrack_min_ablation_round3 import (
    decode_keypoints,
    image_path_for_frame,
    ransac_inlier_keys,
)


def load_cache_by_frame(cache_root, vid):
    out = {}
    for path in sorted(Path(cache_root, "test", f"SNGS-{vid}").glob("frame_*.npz")):
        with np.load(path) as d:
            out[int(d["frame"])] = (str(path), d["kp_hm"].copy(), d["line_hm"].copy())
    return out


def flow_repair_details(base, prev_gray, cur_gray, prev_kp):
    outliers = set(base) - ransac_inlier_keys(base)
    keys = [k for k in outliers if k in prev_kp]
    if not keys or prev_gray is None:
        return [], set(outliers)
    p0 = np.array([[prev_kp[k]["x"] * 1920.0, prev_kp[k]["y"] * 1080.0] for k in keys], dtype=np.float32).reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None, winSize=(21, 21), maxLevel=3)
    details = []
    for key, point, ok, e in zip(keys, p1.reshape(-1, 2), st.reshape(-1), err.reshape(-1)):
        if not ok:
            continue
        x, y = float(point[0] / 1920.0), float(point[1] / 1080.0)
        if -0.1 <= x <= 1.1 and -0.1 <= y <= 1.1:
            details.append(
                {
                    "key": int(key),
                    "prev": (float(prev_kp[key]["x"]), float(prev_kp[key]["y"])),
                    "base": (float(base[key]["x"]), float(base[key]["y"])),
                    "flow": (x, y),
                    "lk_err": float(e),
                }
            )
    return details, set(outliers)


def metric_lookup(path, vid, gid):
    if not path or not Path(path).exists():
        return {}
    for row in csv.DictReader(open(path, newline="")):
        if row.get("video") == vid and row.get("gid") == gid:
            return row
    return {}


def draw_marker(img, xy_norm, color, marker, label=None):
    x, y = int(xy_norm[0] * 1920), int(xy_norm[1] * 1080)
    cv2.drawMarker(img, (x, y), color, marker, 24, 3, cv2.LINE_AA)
    if label is not None:
        cv2.putText(img, str(label), (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)


def fade_background(img, alpha):
    pale = np.full_like(img, 235)
    return cv2.addWeighted(img, float(alpha), pale, 1.0 - float(alpha), 0.0)


def norm_gt_point(gt_keypoints, key):
    item = gt_keypoints.get(key) or gt_keypoints.get(str(key))
    if not item:
        return None
    x, y = float(item["x"]), float(item["y"])
    if x > 2.0 or y > 2.0:
        x, y = x / 1920.0, y / 1080.0
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--frame-scores", required=True)
    ap.add_argument("--video", default="116")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--bg-alpha", type=float, default=0.42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache_by_frame(args.cache_root, args.video)
    zf = zipfile.ZipFile(args.state)
    img_df = pickle.loads(zf.read(f"{args.video}_image.pkl"))
    img_by_id = {str(r["id"]): r for _, r in img_df.iterrows()}

    prev_gray = None
    prev_kp = None
    prev_frame = None
    cases = []
    for frame in sorted(cache):
        idx = frame % 1000000
        if (idx - 1) % args.stride != 0:
            continue
        _path, kp_hm, line_hm = cache[frame]
        base = decode_keypoints(kp_hm, line_hm, "cpu")
        cur_img = cv2.imread(image_path_for_frame(args.video, frame))
        if cur_img is None:
            continue
        cur_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        details, outliers = flow_repair_details(base, prev_gray, cur_gray, prev_kp or {})
        if details:
            cases.append({
                "frame": frame,
                "prev_frame": prev_frame,
                "base": base,
                "prev_kp": prev_kp,
                "details": details,
                "outliers": outliers,
                "cur_img": cur_img,
                "prev_img": cv2.imread(image_path_for_frame(args.video, prev_frame)) if prev_frame else None,
            })
        prev_gray = cur_gray
        prev_kp = base
        prev_frame = frame

    if not cases:
        raise SystemExit("no flow repair case found")

    def fnum(x):
        try:
            return float(x)
        except Exception:
            return None

    def score_case(case):
        gid = f"3{args.video}{case['frame'] % 1000000:06d}"
        metrics = metric_lookup(args.frame_scores, args.video, gid)
        bpt, fpt = fnum(metrics.get("baseline_point")), fnum(metrics.get("flow_point"))
        brp, frp = fnum(metrics.get("baseline_reproj")), fnum(metrics.get("flow_reproj"))
        dpt = (fpt - bpt) if bpt is not None and fpt is not None else 0.0
        drp = (brp - frp) if brp is not None and frp is not None else 0.0
        return len(case["details"]), dpt, drp

    cases = sorted(cases, key=score_case, reverse=True)[: args.topk]
    outputs = []
    payload_cases = []
    for rank, case in enumerate(cases, 1):
        gid = f"3{args.video}{case['frame'] % 1000000:06d}"
        metrics = metric_lookup(args.frame_scores, args.video, gid)
        gt_row = img_by_id.get(str(case["frame"]))
        gt_keypoints = gt_row.get("keypoints", {}) if gt_row is not None else {}
        cur = fade_background(case["cur_img"].copy(), args.bg_alpha)
        prev = fade_background(case["prev_img"].copy(), args.bg_alpha) if case["prev_img"] is not None else np.zeros_like(cur)
        for d in case["details"]:
            key = d["key"]
            draw_marker(cur, d["base"], (0, 0, 255), cv2.MARKER_TILTED_CROSS, key)
            draw_marker(cur, d["flow"], (0, 255, 0), cv2.MARKER_CROSS, key)
            gt = norm_gt_point(gt_keypoints, key)
            if gt:
                draw_marker(cur, gt, (0, 255, 255), cv2.MARKER_STAR, key)
                bx, by = int(d["base"][0] * 1920), int(d["base"][1] * 1080)
                fx, fy = int(d["flow"][0] * 1920), int(d["flow"][1] * 1080)
                gx, gy = int(gt[0] * 1920), int(gt[1] * 1080)
                cv2.arrowedLine(cur, (bx, by), (fx, fy), (0, 255, 0), 3, cv2.LINE_AA, tipLength=0.18)
                cv2.line(cur, (fx, fy), (gx, gy), (0, 255, 255), 2, cv2.LINE_AA)
            if case["prev_img"] is not None:
                draw_marker(prev, d["prev"], (255, 170, 0), cv2.MARKER_DIAMOND, key)
        bpt, fpt = fnum(metrics.get("baseline_point")), fnum(metrics.get("flow_point"))
        bln, fln = fnum(metrics.get("baseline_line")), fnum(metrics.get("flow_line"))
        brp, frp = fnum(metrics.get("baseline_reproj")), fnum(metrics.get("flow_reproj"))
        lines = [
            f"rank={rank} SNGS-{args.video} frame={case['frame']} prev={case['prev_frame']} repaired={len(case['details'])} initial_outliers={len(case['outliers'])}",
            "red X=HRNet outlier, green +=LK recovery, yellow star=GT, blue diamond=previous reference",
        ]
        if bpt is not None and fpt is not None:
            lines.append(f"delta: point {fpt-bpt:+.4f}, line {fln-bln:+.4f}, reproj {frp-brp:+.2f}px")
        for i, text in enumerate(lines):
            cv2.putText(cur, text, (24, 42 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(cur, text, (24, 42 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(prev, text if i == 0 else "previous scored frame reference positions", (24, 42 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(prev, text if i == 0 else "previous scored frame reference positions", (24, 42 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (255, 255, 255), 2, cv2.LINE_AA)
        combo = np.concatenate([prev, cur], axis=1)
        out_img = out_dir / f"SNGS-{args.video}_flow_recovery_top{rank}_{case['frame']}.jpg"
        if rank == 1:
            cv2.imwrite(str(out_dir / f"SNGS-{args.video}_flow_recovery_best_{case['frame']}.jpg"), combo)
        cv2.imwrite(str(out_img), combo)
        outputs.append(str(out_img))
        payload_cases.append({
            "rank": rank,
            "frame": int(case["frame"]),
            "prev_frame": int(case["prev_frame"]) if case["prev_frame"] else None,
            "gid": gid,
            "repaired": len(case["details"]),
            "initial_outliers": len(case["outliers"]),
            "metrics": metrics,
            "details": case["details"],
            "output": str(out_img),
        })

    best = payload_cases[0]
    payload = {"video": args.video, "topk": args.topk, "bg_alpha": args.bg_alpha, "cases": payload_cases}
    (out_dir / "FLOW_RECOVERY_CASE.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "FLOW_RECOVERY_CASE.md").write_text(
        "# Flow Recovery Case\n\n"
        f"- Video: SNGS-{args.video}\n"
        f"- Top cases: {len(payload_cases)}\n"
        f"- Best frame: {best['frame']}\n"
        f"- Previous scored frame: {best['prev_frame']}\n"
        f"- Repaired keypoints: {best['repaired']}\n"
        f"- Initial RANSAC outliers: {best['initial_outliers']}\n"
        f"- Background alpha: {args.bg_alpha}\n"
        "- Legend: red X=current HRNet outlier, green +=LK recovery, yellow star=current GT keypoint, blue diamond=previous reference.\n",
        encoding="utf-8",
    )
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
