"""Render BEV videos comparing baseline/flow/radial calibration on frozen tracks."""
import argparse
import json
import pickle
import zipfile
from pathlib import Path

import cv2
import numpy as np

from inject_outlier_params_to_sam3_state_round3 import bbox_pitch, valid_params


FIELD_X = (-52.5, 52.5)
FIELD_Y = (-34.0, 34.0)


def pitch_to_px(x, y, w, h, margin=24):
    sx = (w - 2 * margin) / (FIELD_X[1] - FIELD_X[0])
    sy = (h - 2 * margin) / (FIELD_Y[1] - FIELD_Y[0])
    px = int(margin + (x - FIELD_X[0]) * sx)
    py = int(h - margin - (y - FIELD_Y[0]) * sy)
    return px, py


def draw_pitch(img):
    h, w = img.shape[:2]
    green = (38, 94, 50)
    white = (220, 235, 225)
    img[:] = green
    x0, y0 = pitch_to_px(FIELD_X[0], FIELD_Y[0], w, h)
    x1, y1 = pitch_to_px(FIELD_X[1], FIELD_Y[1], w, h)
    cv2.rectangle(img, (x0, y1), (x1, y0), white, 2, cv2.LINE_AA)
    xm, _ = pitch_to_px(0, 0, w, h)
    cv2.line(img, (xm, y1), (xm, y0), white, 1, cv2.LINE_AA)
    cx, cy = pitch_to_px(0, 0, w, h)
    cv2.circle(img, (cx, cy), int((w - 48) / 105.0 * 9.15), white, 1, cv2.LINE_AA)
    for sx in [-1, 1]:
        gx = sx * 52.5
        box_x = gx - sx * 16.5
        pxg, _ = pitch_to_px(gx, 0, w, h)
        pxb, _ = pitch_to_px(box_x, 0, w, h)
        _, yt = pitch_to_px(0, 20.16, w, h)
        _, yb = pitch_to_px(0, -20.16, w, h)
        cv2.rectangle(img, (min(pxg, pxb), yt), (max(pxg, pxb), yb), white, 1, cv2.LINE_AA)


def color_for(track_id):
    tid = int(track_id) if track_id == track_id else 0
    rng = np.random.default_rng(tid + 17)
    return tuple(int(x) for x in rng.integers(80, 255, size=3))


def params_for_frame(method_params, image_id):
    if image_id in method_params and valid_params(method_params[image_id]):
        return method_params[image_id]
    keys = sorted(k for k in method_params if k <= image_id and valid_params(method_params[k]))
    if not keys:
        return None
    return method_params[keys[-1]]


def draw_panel(method, rows, method_params, image_id, size):
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    draw_pitch(img)
    params = params_for_frame(method_params, image_id)
    count = bad = 0
    for _, row in rows.iterrows():
        ltwh = row.get("bbox_ltwh")
        bp = bbox_pitch(params, ltwh) if valid_params(params) and ltwh is not None else None
        if not bp:
            bad += 1
            continue
        x, y = bp.get("x_bottom_middle"), bp.get("y_bottom_middle")
        if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
            bad += 1
            continue
        px, py = pitch_to_px(x, y, w, h)
        col = color_for(row.get("track_id", 0))
        cv2.circle(img, (px, py), 4, col, -1, cv2.LINE_AA)
        count += 1
    cv2.putText(img, method, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"points={count} fallback={bad}", (18, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--params-json", required=True)
    ap.add_argument("--video", default="116")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_all = json.load(open(args.params_json, encoding="utf-8"))
    zf = zipfile.ZipFile(args.state)
    det = pickle.loads(zf.read(f"{args.video}.pkl"))
    img = pickle.loads(zf.read(f"{args.video}_image.pkl"))
    methods = ["baseline", "flow", "radial_k1"]
    method_params = {m: params_all[m].get(args.video, {}) for m in methods}

    panel_size = (480, 320)
    out_path = out_dir / f"SNGS-{args.video}_bev_baseline_flow_radial.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (panel_size[0] * 3, panel_size[1] + 42))
    frames = list(img.sort_values("frame").itertuples())
    if args.max_frames:
        frames = frames[: args.max_frames]
    for idx, fr in enumerate(frames):
        image_id = str(fr.id)
        rows = det[det["image_id"].astype(str) == image_id]
        panels = [draw_panel(m, rows, method_params[m], image_id, panel_size) for m in methods]
        canvas = np.zeros((panel_size[1] + 42, panel_size[0] * 3, 3), np.uint8)
        canvas[: panel_size[1], :] = np.concatenate(panels, axis=1)
        cv2.putText(canvas, f"SNGS-{args.video} frame {idx + 1:03d}/{len(frames):03d} image_id={image_id} calibration params held from last stride-20 estimate",
                    (16, panel_size[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        writer.write(canvas)
        if idx in {0, 60, 120, 240, 360, 520, 740}:
            cv2.imwrite(str(out_dir / f"SNGS-{args.video}_bev_frame_{idx + 1:03d}.jpg"), canvas)
    writer.release()
    summary = {
        "video": args.video,
        "frames": len(frames),
        "methods": methods,
        "state": args.state,
        "params_json": args.params_json,
        "output": str(out_path),
        "note": "Tracking/detections are reused from the source state. Calibration params are baseline/flow/radial_k1 from round3_broadtrack_10vid_s20 and held between stride-20 estimates.",
    }
    (out_dir / "BEV_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "BEV_SUMMARY.md").write_text(
        "# BroadTrack BEV Video\n\n"
        f"- Video: SNGS-{args.video}\n"
        f"- Frames rendered: {len(frames)}\n"
        f"- Output: `{out_path}`\n"
        "- Panels: baseline, flow, radial_k1\n"
        "- Detections/tracks: reused from source SAM3 state\n"
        "- Calibration: round3 stride-20 params, held until next estimate\n",
        encoding="utf-8",
    )
    print(out_path)


if __name__ == "__main__":
    main()
