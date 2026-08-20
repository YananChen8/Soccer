#!/usr/bin/env python
import csv
import importlib.util
import json
import math
import os
import numpy as np

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
    H, _, _ = pa.ransac_homography(np.asarray(src), np.asarray(dst))
    return H


def folded_angle(v, axis):
    if axis == "y":
        a = math.degrees(math.atan2(float(v[0]), float(v[1])))
    else:
        a = math.degrees(math.atan2(float(v[1]), float(v[0])))
    while a > 90:
        a -= 180
    while a < -90:
        a += 180
    return a


rows = list(csv.DictReader(open(os.path.join(pa.REPORT, pa.RUN_DIRS[0], "test_frame_scores.csv"))))
keys = []
for r in rows:
    k = (r["video"], r["frame"])
    if k not in keys:
        keys.append(k)
Hs = []
for k in keys:
    H = homography_for(*k)
    if H is not None:
        Hs.append(H)
print("Hs", len(Hs))

variants = {
    "bottom_to_center_vs_y": ([[0.5, 1.0], [0.5, 0.5]], "y"),
    "lower_to_center_vs_y": ([[0.5, 0.85], [0.5, 0.5]], "y"),
    "center_to_top_vs_y": ([[0.5, 0.5], [0.5, 0.0]], "y"),
    "lower_to_center_vs_x": ([[0.5, 0.85], [0.5, 0.5]], "x"),
    "left_to_right_vs_y": ([[0.25, 0.5], [0.75, 0.5]], "y"),
    "left_to_right_vs_x": ([[0.25, 0.5], [0.75, 0.5]], "x"),
}
for name, (pts, axis) in variants.items():
    vals = []
    for H in Hs:
        p = pa.project(H, np.asarray(pts, dtype=np.float64))
        v = p[1] - p[0]
        if np.isfinite(v).all() and np.linalg.norm(v) > 1e-6:
            vals.append(folded_angle(v, axis))
    if vals:
        arr = np.asarray(vals)
        print(name, np.percentile(arr, [5, 25, 50, 75, 95]).round(2).tolist())
