"""Inject round3 calibration params into a frozen SAM3 SoccerNetGS state.

Only image camera parameters and detection bbox_pitch are changed. Image-space
detections, track ids, roles, teams, jerseys, masks, and embeddings are copied.
"""
import argparse
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

SNG = Path("/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate")
sys.path.insert(0, str(SNG))
sys.path.insert(0, str(SNG / "plugins/calibration/nbjw_calib"))
sys.path.insert(0, str(SNG / "plugins/calibration"))

from nbjw_calib.utils.utils_calib import pan_tilt_roll_to_orientation as PTR
from sn_gamestate.structured_calibration.weighted_solver import H_from_params


def valid_params(params):
    return (
        isinstance(params, dict)
        and "pan_degrees" in params
        and "tilt_degrees" in params
        and "roll_degrees" in params
        and "x_focal_length" in params
        and "y_focal_length" in params
        and "principal_point" in params
        and "position_meters" in params
    )


def ltrb_of(ltwh):
    l, t, w, h = ltwh
    return float(l), float(t), float(l + w), float(t + h)


def bbox_pitch(params, ltwh):
    hmat = H_from_params(params, PTR)
    if hmat is None:
        return None
    try:
        inv = np.linalg.inv(hmat)
    except np.linalg.LinAlgError:
        return None
    l, _t, r, b = ltrb_of(ltwh)
    out = {}
    for name, (u, v) in {
        "bottom_left": (l, b),
        "bottom_right": (r, b),
        "bottom_middle": ((l + r) / 2.0, b),
    }.items():
        q = inv @ np.array([u, v, 1.0], dtype=float)
        if abs(q[2]) < 1e-9:
            return None
        out[f"x_{name}"] = float(q[0] / q[2])
        out[f"y_{name}"] = float(q[1] / q[2])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-state", required=True)
    ap.add_argument("--params-json", required=True)
    ap.add_argument("--adapter", default="stgcn_k15_lr1e4_outlier_only")
    ap.add_argument("--out-state", required=True)
    ap.add_argument("--videos", nargs="+", default=["116", "117", "118"])
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    params_all = json.load(open(args.params_json, encoding="utf-8"))[args.adapter]
    source = zipfile.ZipFile(args.source_state)
    out_path = Path(args.out_state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED)
    names = set(source.namelist())
    stats = {"videos": {}, "adapter": args.adapter, "source_state": args.source_state}

    if "summary.json" in names:
        out.writestr("summary.json", source.read("summary.json"))

    for vid in args.videos:
        det = pickle.loads(source.read(f"{vid}.pkl"))
        img = pickle.loads(source.read(f"{vid}_image.pkl"))
        vid_params = params_all.get(vid, {})
        par_map = {}
        img_new = img.copy()
        replaced_params = 0
        for idx, row in img.iterrows():
            image_id = str(row["id"])
            new_params = vid_params.get(image_id)
            if valid_params(new_params):
                img_new.at[idx, "parameters"] = new_params
                par_map[image_id] = new_params
                replaced_params += 1
            else:
                par_map[image_id] = row.get("parameters")

        det_new = det.copy()
        new_pitch = []
        reproj = fallback = 0
        for _, row in det.iterrows():
            image_id = str(row.get("image_id"))
            params = par_map.get(image_id)
            ltwh = row.get("bbox_ltwh")
            bp = None
            if valid_params(params) and ltwh is not None:
                bp = bbox_pitch(params, ltwh)
            if bp is None:
                bp = row.get("bbox_pitch")
                fallback += 1
            else:
                reproj += 1
            new_pitch.append(bp)
        det_new["bbox_pitch"] = new_pitch

        out.writestr(f"{vid}.pkl", pickle.dumps(det_new))
        out.writestr(f"{vid}_image.pkl", pickle.dumps(img_new))
        stats["videos"][vid] = {
            "images": int(len(img)),
            "detections": int(len(det)),
            "image_params_replaced": int(replaced_params),
            "bbox_pitch_reprojected": int(reproj),
            "bbox_pitch_fallback": int(fallback),
        }
        print(vid, stats["videos"][vid], flush=True)

    for name in names:
        if name == "summary.json" or any(name == f"{v}.pkl" or name == f"{v}_image.pkl" for v in args.videos):
            continue
        out.writestr(name, source.read(name))
    out.close()
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("wrote", out_path)
    print("manifest", args.manifest)


if __name__ == "__main__":
    main()
