#!/usr/bin/env python3
"""Score cached primitive params.json into the Round3 four-metric table."""
from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

from sn_gamestate.structured_calibration.cached_primitive_eval_round3 import (
    aggregate_results,
    fmt,
    score_job,
    smoothness,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--data-root", default="datasets/SoccerNetGS/valid")
    ap.add_argument("--nproc", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root)
    params = json.loads((root / "params.json").read_text())
    methods = list(params)
    jobs = [(args.data_root, vid, method, params[method][vid]) for method in methods for vid in params[method]]
    print(f"score_jobs={len(jobs)} methods={methods}", flush=True)
    with Pool(min(args.nproc, len(jobs))) as pool:
        parts = pool.map(score_job, jobs)

    smooth = {m: [smoothness(params[m][vid]) for vid in params[m]] for m in methods}
    results = aggregate_results(params, smooth, parts)
    (root / "result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "| method | point | line | reproj | smooth_mean | smooth_p95 | frames |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for method in methods:
        r = results[method]
        lines.append(
            f"| {method} | {fmt(r['point_acc'])} | {fmt(r['line_acc'])} | {fmt(r['reproj_mean'], 2)} | "
            f"{fmt(r['smoothness_mean'], 2)} | {fmt(r['smoothness_p95'], 2)} | {r['n_frames']} |"
        )
        rows.append({
            "method": method,
            "point": r["point_acc"],
            "line": r["line_acc"],
            "reproj": r["reproj_mean"],
            "smooth_mean": r["smoothness_mean"],
            "smooth_p95": r["smoothness_p95"],
            "frames": r["n_frames"],
        })
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(root / "results.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print((root / "RESULTS.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
