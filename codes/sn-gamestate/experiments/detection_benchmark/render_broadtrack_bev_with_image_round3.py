"""Render original frame + BEV reference/baseline/flow/radial panels."""
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


def pitch_to_px(x, y, w, h, margin=18):
    sx = (w - 2 * margin) / (FIELD_X[1] - FIELD_X[0])
    sy = (h - 2 * margin) / (FIELD_Y[1] - FIELD_Y[0])
    return int(margin + (x - FIELD_X[0]) * sx), int(h - margin - (y - FIELD_Y[0]) * sy)


def draw_pitch(img):
    h, w = img.shape[:2]
    img[:] = (36, 88, 48)
    white = (225, 235, 225)
    x0, y0 = pitch_to_px(FIELD_X[0], FIELD_Y[0], w, h)
    x1, y1 = pitch_to_px(FIELD_X[1], FIELD_Y[1], w, h)
    cv2.rectangle(img, (x0, y1), (x1, y0), white, 2, cv2.LINE_AA)
    xm, _ = pitch_to_px(0, 0, w, h)
    cv2.line(img, (xm, y1), (xm, y0), white, 1, cv2.LINE_AA)
    cx, cy = pitch_to_px(0, 0, w, h)
    cv2.circle(img, (cx, cy), int((w - 36) / 105.0 * 9.15), white, 1, cv2.LINE_AA)
    for sx in [-1, 1]:
        goal_x = sx * 52.5
        box_x = goal_x - sx * 16.5
        pxg, _ = pitch_to_px(goal_x, 0, w, h)
        pxb, _ = pitch_to_px(box_x, 0, w, h)
        _, yt = pitch_to_px(0, 20.16, w, h)
        _, yb = pitch_to_px(0, -20.16, w, h)
        cv2.rectangle(img, (min(pxg, pxb), yt), (max(pxg, pxb), yb), white, 1, cv2.LINE_AA)


def color_for(track_id):
    tid = int(track_id) if track_id == track_id else 0
    rng = np.random.default_rng(tid + 19)
    return tuple(int(x) for x in rng.integers(80, 255, size=3))


def params_for_frame(method_params, image_id):
    if image_id in method_params and valid_params(method_params[image_id]):
        return method_params[image_id]
    keys = sorted(k for k in method_params if k <= image_id and valid_params(method_params[k]))
    return method_params[keys[-1]] if keys else None


def point_from_bbox_pitch(bp):
    if isinstance(bp, dict):
        return bp.get("x_bottom_middle"), bp.get("y_bottom_middle")
    return None, None


def draw_bev_panel(name, rows, params=None, size=(360, 238), source=False):
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    draw_pitch(img)
    count = bad = 0
    for _, row in rows.iterrows():
        if source:
            x, y = point_from_bbox_pitch(row.get("bbox_pitch"))
        else:
            bp = bbox_pitch(params, row.get("bbox_ltwh")) if valid_params(params) else None
            x, y = point_from_bbox_pitch(bp)
        if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
            bad += 1
            continue
        px, py = pitch_to_px(float(x), float(y), w, h)
        cv2.circle(img, (px, py), 3, color_for(row.get("track_id", 0)), -1, cv2.LINE_AA)
        count += 1
    cv2.putText(img, name, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"n={count} bad={bad}", (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_source_frame(path, rows, size=(720, 405)):
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((1080, 1920, 3), np.uint8)
    sx, sy = size[0] / img.shape[1], size[1] / img.shape[0]
    img = cv2.resize(img, size)
    for _, row in rows.iterrows():
        try:
            l, t, w, h = [float(x) for x in row.get("bbox_ltwh")]
        except Exception:
            continue
        p1 = int(l * sx), int(t * sy)
        p2 = int((l + w) * sx), int((t + h) * sy)
        cv2.rectangle(img, p1, p2, color_for(row.get("track_id", 0)), 1, cv2.LINE_AA)
    cv2.putText(img, "source image + reused tracks", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
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
    img_df = pickle.loads(zf.read(f"{args.video}_image.pkl"))
    methods = ["baseline", "flow", "radial_k1"]
    method_params = {m: params_all[m].get(args.video, {}) for m in methods}

    source_size = (720, 405)
    bev_size = (360, 238)
    frame_w = 720
    frame_h = 405 + 238 * 2 + 40
    out_path = out_dir / f"SNGS-{args.video}_image_ref_baseline_flow_radial_bev.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (frame_w, frame_h))
    frames = list(img_df.sort_values("frame").itertuples())
    if args.max_frames:
        frames = frames[: args.max_frames]

    for idx, fr in enumerate(frames):
        image_id = str(fr.id)
        rows = det[det["image_id"].astype(str) == image_id]
        top = draw_source_frame(fr.file_path, rows, source_size)
        panels = [
            draw_bev_panel("ref/source bbox_pitch", rows, source=True, size=bev_size),
            draw_bev_panel("baseline", rows, params_for_frame(method_params["baseline"], image_id), bev_size),
            draw_bev_panel("flow", rows, params_for_frame(method_params["flow"], image_id), bev_size),
            draw_bev_panel("radial_k1", rows, params_for_frame(method_params["radial_k1"], image_id), bev_size),
        ]
        row1 = np.concatenate(panels[:2], axis=1)
        row2 = np.concatenate(panels[2:], axis=1)
        canvas = np.zeros((frame_h, frame_w, 3), np.uint8)
        canvas[0:405] = top
        canvas[405:643] = row1
        canvas[643:881] = row2
        cv2.putText(canvas, f"SNGS-{args.video} frame {idx+1:03d}/{len(frames):03d} image_id={image_id}",
                    (12, frame_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
        writer.write(canvas)
        if idx in {0, 60, 120, 240, 360, 520, 740}:
            cv2.imwrite(str(out_dir / f"SNGS-{args.video}_image_bev_frame_{idx+1:03d}.jpg"), canvas)
    writer.release()
    (out_dir / "IMAGE_BEV_SUMMARY.md").write_text(
        "# Image + BEV comparison\n\n"
        f"- Video: SNGS-{args.video}\n"
        f"- Frames: {len(frames)}\n"
        f"- Output: `{out_path}`\n"
        "- Top panel: original frame with reused track boxes.\n"
        "- BEV panels: source bbox_pitch reference, baseline, flow, radial_k1.\n"
        "- Note: source bbox_pitch is a reference from the reused state, not confirmed manual GT.\n",
        encoding="utf-8",
    )
    print(out_path)


if __name__ == "__main__":
    main()
