import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(
    "outputs/gsr/temporal_hrnet/temporal_calib_results_hub/"
    "full_finetune_temporal_nbjw_k3_official_aux_20260701/report_eval_visual_20260701"
)
RUN_DIRS = sorted(ROOT.glob("test_stride5_scatter_reproj25_fullft_offaux_*_k3"))
ANGLE_COL = "signed_angle_to_midline_deg"
REPROJ_MAX = 25.0
ANCHOR_RUN = "fullft_offaux_last_motion_k3"
ANCHOR_VIDEO = "120"
ANCHOR_FRAME = "000501"


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def finite_float(value):
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def get_anchor_center():
    fallback = None
    for run_dir in RUN_DIRS:
        frame_csv = run_dir / "test_frame_scores.csv"
        if not frame_csv.exists():
            continue
        for r in read_rows(frame_csv):
            if r.get("video") != ANCHOR_VIDEO or r.get("frame") != ANCHOR_FRAME:
                continue
            angle = finite_float(r.get(ANGLE_COL))
            if angle is None:
                continue
            if r.get("run") == "baseline":
                fallback = angle
            if r.get("run") == ANCHOR_RUN:
                return angle, {"run": r.get("run"), "video": r.get("video"), "frame": r.get("frame")}
    if fallback is not None:
        return fallback, {"run": "baseline", "video": ANCHOR_VIDEO, "frame": ANCHOR_FRAME}
    raise RuntimeError("anchor frame angle not found")


def centered_angle(row, global_center):
    angle = finite_float(row.get(ANGLE_COL))
    if angle is None:
        return None
    x = angle - global_center
    # The existing angle is an orientation relative to the midfield line, so +90
    # and -90 are the same direction with opposite representation. Use 180 deg
    # periodic wrapping before plotting.
    while x > 90:
        x -= 180
    while x < -90:
        x += 180
    return x


def write_points_csv(rows, global_center, out_csv):
    out = []
    for r in rows:
        reproj = finite_float(r.get("reproj_mean"))
        angle = centered_angle(r, global_center)
        if reproj is None or angle is None or reproj >= REPROJ_MAX:
            continue
        item = dict(r)
        item["centered_angle_deg"] = f"{angle:.6f}"
        item["global_angle_center_deg"] = f"{global_center:.6f}"
        out.append(item)
    if out:
        fields = list(out[0].keys())
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out)
    return out


def plot_scatter(points, run_name, out_png):
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    colors = {"baseline": "#222222", run_name: "#d62728"}
    labels = {"baseline": "baseline", run_name: run_name}
    for run in ["baseline", run_name]:
        xs = [float(r["centered_angle_deg"]) for r in points if r["run"] == run]
        ys = [float(r["reproj_mean"]) for r in points if r["run"] == run]
        if not xs:
            continue
        ax.scatter(xs, ys, s=9, alpha=0.42 if run == "baseline" else 0.58, c=colors[run], label=f"{labels[run]} (n={len(xs)})", linewidths=0)
    ax.axvline(0, color="#555555", lw=1.0, linestyle="--", alpha=0.8)
    ax.set_xlim(-90, 90)
    ax.set_ylim(0, REPROJ_MAX)
    ax.set_xlabel("global-centered signed view angle (deg)")
    ax.set_ylabel("reproj mean")
    ax.set_title(f"Angle vs reproj, reproj < {REPROJ_MAX:g}: {run_name}")
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_heatmap(points, run_name, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), dpi=180, sharex=True, sharey=True)
    vmax = None
    hists = []
    for run in ["baseline", run_name]:
        xs = np.array([float(r["centered_angle_deg"]) for r in points if r["run"] == run], dtype=float)
        ys = np.array([float(r["reproj_mean"]) for r in points if r["run"] == run], dtype=float)
        hist, xedges, yedges = np.histogram2d(xs, ys, bins=[36, 25], range=[[-90, 90], [0, REPROJ_MAX]])
        hists.append((run, hist, xedges, yedges))
        vmax = max(vmax or 0, float(hist.max()))
    for ax, (run, hist, xedges, yedges) in zip(axes, hists):
        im = ax.imshow(
            hist.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=vmax,
        )
        ax.axvline(0, color="white", lw=1.0, linestyle="--", alpha=0.9)
        ax.set_title(run)
        ax.set_xlabel("global-centered signed view angle (deg)")
        ax.grid(False)
    axes[0].set_ylabel("reproj mean")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, label="frame count")
    fig.suptitle(f"Frequency heatmap, reproj < {REPROJ_MAX:g}: {run_name}")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    summary = []
    global_center, anchor = get_anchor_center()
    for run_dir in RUN_DIRS:
        frame_csv = run_dir / "test_frame_scores.csv"
        if not frame_csv.exists():
            continue
        rows = read_rows(frame_csv)
        run_names = sorted({r["run"] for r in rows if r.get("run") != "baseline"})
        if len(run_names) != 1:
            continue
        run_name = run_names[0]
        out_dir = run_dir / "angle_reproj_scatter_global_centered"
        out_dir.mkdir(parents=True, exist_ok=True)
        points = write_points_csv(rows, global_center, out_dir / "angle_reproj_points_global_centered.csv")
        plot_scatter(points, run_name, out_dir / "angle_reproj_scatter_global_centered.png")
        plot_heatmap(points, run_name, out_dir / "angle_reproj_frequency_heatmap_global_centered.png")
        summary.append(
            {
                "run_dir": run_dir.name,
                "run_name": run_name,
                "source_csv": str(frame_csv),
                "out_dir": str(out_dir),
                "rows_in": len(rows),
                "rows_reproj_lt_25": len(points),
                "global_center_deg": global_center,
                "anchor": anchor,
                "angle_definition": f"centered_angle_deg = period180({ANGLE_COL} - anchor {ANGLE_COL}); anchor is {ANCHOR_RUN} SNGS-{ANCHOR_VIDEO} frame {ANCHOR_FRAME}",
            }
        )
    (ROOT / "global_centered_angle_replot_summary_20260701.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
