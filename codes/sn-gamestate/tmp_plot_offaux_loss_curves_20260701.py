import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701")
OUT = ROOT / "report_eval_visual_20260701" / "loss_curves"
RUNS = [
    "fullft_offaux_last_motion_k3",
    "fullft_offaux_last_nomotion_k3",
    "fullft_offaux_stage1_motion_k3",
    "fullft_offaux_stage1_nomotion_k3",
]
PAT = re.compile(
    r"epoch=(?P<epoch>\d+) step=(?P<step>\d+) "
    r"loss=(?P<loss>[0-9.eE+-]+) heat=(?P<heat>[0-9.eE+-]+) "
    r"peak=(?P<peak>[0-9.eE+-]+) motion=(?P<motion>[0-9.eE+-]+) "
    r"peak_term=(?P<peak_term>[0-9.eE+-]+) motion_term=(?P<motion_term>[0-9.eE+-]+) "
    r"fg=(?P<fg>[0-9.eE+-]+) frames/s=(?P<fps>[0-9.eE+-]+)"
)


def parse_run(run):
    rows = []
    log = ROOT / run / "logs" / "train.log"
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = PAT.search(line)
        if not m:
            continue
        row = {"run": run}
        row.update({k: float(v) for k, v in m.groupdict().items()})
        row["epoch"] = int(row["epoch"])
        row["step"] = int(row["step"])
        rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    by_run = {}
    for run in RUNS:
        by_run[run] = parse_run(run)
        rows.extend(by_run[run])
    fields = ["run", "epoch", "step", "loss", "heat", "peak", "motion", "peak_term", "motion_term", "fg", "fps"]
    with (OUT / "loss_curves_long.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    metrics = ["loss", "heat", "peak", "motion", "peak_term", "motion_term", "fg", "fps"]
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
    for ax, metric in zip(axes.ravel(), metrics):
        for run in RUNS:
            xs = [r["step"] for r in by_run[run]]
            ys = [r[metric] for r in by_run[run]]
            ax.plot(xs, ys, linewidth=1.4, label=run.replace("fullft_offaux_", ""))
        ax.set_title(metric)
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("step")
    axes[-1, 1].set_xlabel("step")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "loss_curves_all_metrics.png", dpi=180)
    plt.close(fig)

    for metric in metrics:
        plt.figure(figsize=(9, 5))
        for run in RUNS:
            xs = [r["step"] for r in by_run[run]]
            ys = [r[metric] for r in by_run[run]]
            plt.plot(xs, ys, linewidth=1.6, label=run.replace("fullft_offaux_", ""))
        plt.title(metric)
        plt.xlabel("step")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT / f"{metric}_curve.png", dpi=180)
        plt.close()

    (OUT / "README.md").write_text(
        "Parsed from each run's logs/train.log. Curves show logged training values only; warnings/invalid GT frames are excluded from the CSV.\n",
        encoding="utf-8",
    )
    print(OUT)


if __name__ == "__main__":
    main()
