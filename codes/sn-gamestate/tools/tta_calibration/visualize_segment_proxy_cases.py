#!/usr/bin/env python3
"""Render selected segment-proxy audit cases."""
import csv
import json
from pathlib import Path

import tta_proxy_audit as old


ROOT = Path(".")
OUT = ROOT / "outputs/tta_calib/segment_proxy_audit"
CASES = OUT / "tables/case_manifest.csv"
FIG = OUT / "overlays"

RUNS = {
    "full49_safe": ROOT / "outputs/tta_calib/tta_v1_camera/fast_full49_safe_eval",
    "full49_strict002": ROOT / "outputs/tta_calib/tta_v1_camera/fast_full49_strict002_eval",
    "subset8_safe": ROOT / "outputs/tta_calib/tta_v1_camera/fast_subset8_safe_eval",
    "subset8_strict002": ROOT / "outputs/tta_calib/tta_v1_camera/fast_subset8_strict002_eval",
    "subset8_per_frame_k1": ROOT / "outputs/tta_calib/tta_v1_camera/fast_subset8_k1_strict002_eval",
    "subset8_video_shared_k1": ROOT / "outputs/tta_calib/tta_v2_video_k1/fast_subset8_video_k1_eval",
}


def load(run):
    data = json.load(open(RUNS[run] / "params.json"))
    return data["baseline_raw"], data["tta_v1_fast"]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    params = {}
    made = 0
    with CASES.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            run, vid, gid = r["run"], str(r["video"]), str(r["frame"])
            if run not in params:
                params[run] = load(run)
            raw, tta = params[run]
            out_dir = FIG / r["case_group"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{int(r['rank']):02d}_{run}_SNGS-{vid}_{gid}_dacc_{float(r['delta_acc']):+.4f}.jpg"
            if old.draw_overlay(run, {"video": vid, "gid": gid, "delta_acc": r["delta_acc"]}, raw[vid].get(gid), tta[vid].get(gid), out):
                made += 1
    print(f"rendered_overlays: {made}")
    print("overlay_dir: outputs/tta_calib/segment_proxy_audit/overlays")


if __name__ == "__main__":
    main()

