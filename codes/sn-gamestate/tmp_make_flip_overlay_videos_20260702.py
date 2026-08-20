import argparse
import copy
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import tmp_official_aux_report_eval_visual_20260701 as ref
import tmp_true_tta_nbjw_20260702 as tta
from sn_calibration_baseline.camera import Camera


SEGMENTS = [
    [(0, 0), (0, 68)], [(105, 0), (105, 68)], [(0, 0), (105, 0)], [(0, 68), (105, 68)],
    [(52.5, 0), (52.5, 68)],
    [(0, 13.85), (16.5, 13.85)], [(0, 54.15), (16.5, 54.15)], [(16.5, 13.85), (16.5, 54.15)],
    [(0, 24.85), (5.5, 24.85)], [(0, 43.15), (5.5, 43.15)], [(5.5, 24.85), (5.5, 43.15)],
    [(105, 13.85), (88.5, 13.85)], [(105, 54.15), (88.5, 54.15)], [(88.5, 13.85), (88.5, 54.15)],
    [(105, 24.85), (99.5, 24.85)], [(105, 43.15), (99.5, 43.15)], [(99.5, 24.85), (99.5, 43.15)],
]


def arc(cx, cy, r, a0, a1, n=80):
    aa = np.linspace(np.deg2rad(a0), np.deg2rad(a1), n)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in aa]


POLYLINES = [[a, b] for a, b in SEGMENTS] + [
    arc(52.5, 34, 9.15, 0, 360),
    arc(11, 34, 9.15, -52, 52),
    arc(94, 34, 9.15, 128, 232),
]


def fmt_metric(v):
    return "NA" if v is None or not np.isfinite(v) else f"{float(v):.2f}"


def pitch_to_bev(pt, w=960, h=270):
    x, y = pt
    pad_x, pad_y = 30, 20
    sx = (w - 2 * pad_x) / 105.0
    sy = (h - 2 * pad_y) / 68.0
    return int(round(pad_x + x * sx)), int(round(pad_y + y * sy))


def decode_keypoints(hm):
    try:
        return tta.decode_anchor_keypoints(hm.detach(), 0.05)
    except Exception:
        return {}


def project_polyline(params, polyline):
    if not params:
        return []
    cam = Camera(iwidth=960, iheight=540)
    cam.from_json_parameters(params)
    pts = []
    for x, y in polyline:
        p = cam.project_point(np.asarray([x, y, 0.0], dtype=float))
        if p[2] > 0 and np.isfinite(p[:2]).all():
            pts.append((int(round(p[0])), int(round(p[1]))))
        else:
            pts.append(None)
    return pts


def draw_projection(img_rgb, params, title, color, keypoints=None):
    img = np.asarray(img_rgb).copy()
    for poly in POLYLINES:
        pts = project_polyline(params, poly)
        run = []
        for p in pts:
            if p is not None and -200 <= p[0] <= 1160 and -200 <= p[1] <= 740:
                run.append(p)
            else:
                if len(run) >= 2:
                    cv2.polylines(img, [np.asarray(run, np.int32)], False, color, 2, cv2.LINE_AA)
                run = []
        if len(run) >= 2:
            cv2.polylines(img, [np.asarray(run, np.int32)], False, color, 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (960, 34), (0, 0, 0), -1)
    cv2.putText(img, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    for kid, kp in (keypoints or {}).items():
        x, y = int(round(kp["x"])), int(round(kp["y"]))
        if 0 <= x < 960 and 0 <= y < 540:
            cv2.circle(img, (x, y), 4, (255, 220, 40), -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 6, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def draw_bev(title, color, keypoints=None):
    bev = np.full((270, 960, 3), (28, 92, 50), dtype=np.uint8)
    for poly in POLYLINES:
        pts = np.asarray([pitch_to_bev(p) for p in poly], np.int32)
        if len(pts) >= 2:
            cv2.polylines(bev, [pts], False, (235, 245, 235), 2, cv2.LINE_AA)
    for kid, kp in (keypoints or {}).items():
        idx = int(kid) - 1
        if 0 <= idx < len(tta.keypoint_world_coords_2D):
            x, y = pitch_to_bev(tta.keypoint_world_coords_2D[idx])
            cv2.circle(bev, (x, y), 5, color, -1, cv2.LINE_AA)
            cv2.circle(bev, (x, y), 7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(bev, (0, 0), (960, 28), (0, 0, 0), -1)
    cv2.putText(bev, title, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return bev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    frames_root, data_root = ref.get_split_paths("test")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kp_swap = tta.keypoint_swap()
    kp_model, line_model = ref.base.load_hrnets(device)
    line_model.eval()
    base_state = copy.deepcopy(kp_model.state_dict())
    tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    tta_args = types.SimpleNamespace(
        steps=1, lr=1e-5, anchor_weight=0.05, peak_weight=0.0,
        pseudo_conf_threshold=0.05, pseudo_ransac_px=20.0,
        pseudo_sigma=2.0, outlier_weight=0.001,
        temporal_sigma=2.0, temporal_gate_px=30.0, temporal_weight=0.2,
    )

    for video in [str(v).replace("SNGS-", "") for v in args.videos]:
        files = sorted((frames_root / f"SNGS-{video}" / "img1").glob("*.jpg"))
        gt = ref.base.load_gt_lines_for_video(str(data_root), video)
        id_map = ref.image_id_map(data_root, video)
        writer = cv2.VideoWriter(
            str(out_dir / f"SNGS-{video}_baseline_vs_flip.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            12.0 if args.stride == 1 else 4.0,
            (1920, 810),
        )
        rows = []
        for idx, path in enumerate(files):
            if idx % args.stride != 0:
                continue
            gid = id_map.get(path.stem, f"3{video}{path.stem}")
            pil = Image.open(path).convert("RGB").resize((960, 540))
            img = tfm(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                line_hm = line_model(img)
                kp_model.load_state_dict(base_state)
                kp_model.eval()
                raw_hm = kp_model(img)
            kp_model.load_state_dict(base_state)
            tta.adapt_model(kp_model, img, kp_swap, "flip_consistency", tta_args)
            with torch.no_grad():
                flip_hm = kp_model(img)

            if gid not in gt:
                continue
            raw = ref.score_hm(raw_hm, line_hm, gt[gid])
            flip = ref.score_hm(flip_hm, line_hm, gt[gid])
            raw_kp = decode_keypoints(raw_hm)
            flip_kp = decode_keypoints(flip_hm)
            raw_title = f"Baseline SNGS-{video} frame {path.stem} MRE={fmt_metric(raw['reproj_mean'])} kp={len(raw_kp)}"
            flip_title = f"Flip-Consistency MRE={fmt_metric(flip['reproj_mean'])} kp={len(flip_kp)}"
            left = draw_projection(pil, raw["params"], raw_title, (255, 64, 64), raw_kp)
            right = draw_projection(pil, flip["params"], flip_title, (64, 255, 64), flip_kp)
            bev_left = draw_bev("Baseline BEV: decoded semantic keypoints", (255, 64, 64), raw_kp)
            bev_right = draw_bev("Flip BEV: decoded semantic keypoints", (64, 255, 64), flip_kp)
            panel = np.concatenate([
                np.concatenate([left, right], axis=1),
                np.concatenate([bev_left, bev_right], axis=1),
            ], axis=0)
            writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            rows.append((path.stem, raw["reproj_mean"], flip["reproj_mean"]))
        writer.release()
        print(f"video={video} frames={len(rows)} out={out_dir / f'SNGS-{video}_baseline_vs_flip.mp4'}", flush=True)


if __name__ == "__main__":
    main()
