#!/usr/bin/env python
import csv
import importlib.util
import json
import math
import os
import sys
import numpy as np

sys.path.insert(0, "/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/plugins/calibration/sn_calibration_baseline")
from camera import Camera, rotation_matrix_to_pan_tilt_roll

spec = importlib.util.spec_from_file_location("pa", "tmp_replot_stride5_scatter_physical_angle_20260701.py")
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)


def homography_for(video, frame):
    path = pa.frame_json(video, frame)
    if not os.path.exists(path):
        return None
    data = json.load(open(path))
    img_lines, world_lines = {}, {}
    for name, pts in data.items():
        if name not in pa.PITCH_LINES or not isinstance(pts, list) or len(pts) < 2:
            continue
        xy = [(float(p["x"]), float(p["y"])) for p in pts if "x" in p and "y" in p]
        il = pa.fit_line(xy)
        wl = pa.pitch_line(name)
        if il is not None and wl is not None:
            img_lines[name] = il
            world_lines[name] = wl
    src, dst = [], []
    names = sorted(img_lines)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pi = pa.intersect(img_lines[a], img_lines[b])
            pw = pa.intersect(world_lines[a], world_lines[b])
            if pi is None or pw is None:
                continue
            if not (-0.5 <= pi[0] <= 1.5 and -0.5 <= pi[1] <= 1.5):
                continue
            src.append(pi)
            dst.append(pw)
    if len(src) < 4:
        return None
    H_img_to_pitch, _, _ = pa.ransac_homography(np.asarray(src), np.asarray(dst))
    if H_img_to_pitch is None:
        return None
    # Convert normalized image coordinates to 1920x1080 pixels.
    S = np.array([[1.0 / 1920.0, 0, 0], [0, 1.0 / 1080.0, 0], [0, 0, 1.0]])
    H_pix_to_pitch = H_img_to_pitch @ S
    return np.linalg.inv(H_pix_to_pitch)


def signed_to_midline_from_pan(pan_deg):
    # Middle line direction in this camera convention has two equivalent headings.
    cands = []
    for ref in (0.0, 180.0, -180.0, 90.0, -90.0):
        d = pan_deg - ref
        while d > 180:
            d -= 360
        while d < -180:
            d += 360
        if abs(d) <= 90:
            cands.append(d)
    return min(cands, key=lambda x: abs(x)) if cands else pan_deg


rows = list(csv.DictReader(open(os.path.join(pa.REPORT, pa.RUN_DIRS[0], "test_frame_scores.csv"))))
keys = []
for r in rows:
    k = (r["video"], r["frame"])
    if k not in keys:
        keys.append(k)

pans, signed0, signed90 = [], [], []
for video, frame in keys:
    H = homography_for(video, frame)
    if H is None:
        continue
    cam = Camera(1920, 1080)
    ok = cam.from_homography(H)
    if not ok:
        continue
    pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(cam.rotation)
    pd = math.degrees(pan)
    pans.append(pd)
    d0 = pd
    while d0 > 90:
        d0 -= 180
    while d0 < -90:
        d0 += 180
    signed0.append(d0)
    ds = []
    for ref in (90.0, -90.0):
        d = pd - ref
        while d > 180:
            d -= 360
        while d < -180:
            d += 360
        ds.append(d)
    signed90.append(min(ds, key=lambda x: abs(x)))

for name, vals in [("pan", pans), ("signed_to_0_axis", signed0), ("signed_to_90_axis", signed90)]:
    if vals:
        arr = np.asarray(vals)
        print(name, len(arr), np.percentile(arr, [5, 25, 50, 75, 95]).round(2).tolist())
