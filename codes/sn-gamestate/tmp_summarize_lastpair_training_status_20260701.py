import csv
import math
import statistics
from pathlib import Path


ROOT = Path("outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k5_k15_motion_lastpair_fast_20260701")
FIELDS = [
    "total_loss",
    "heat_loss",
    "peak_weighted",
    "motion_weighted",
    "peak_raw",
    "motion_raw",
    "motion_valid_pair_ratio",
    "mean_gt_motion_norm",
    "mean_pred_motion_norm",
    "mean_motion_residual_norm",
    "fps",
]


def median(rows, key):
    vals = [r[key] for r in rows if key in r and math.isfinite(r[key])]
    return statistics.median(vals) if vals else float("nan")


for d in sorted(ROOT.glob("fullft_*")):
    p = d / "step_losses.csv"
    print("====", d.name)
    if not p.exists():
        print("NO step_losses.csv")
        continue

    rows = []
    with p.open(newline="") as f:
        for r in csv.DictReader(f):
            rr = {k: float(r[k]) for k in FIELDS if k in r and r[k] != ""}
            rr["epoch"] = int(float(r["epoch"]))
            rr["step"] = int(float(r["step"]))
            rr["frames"] = int(float(r["frames"]))
            rows.append(rr)

    print("rows", len(rows), "step", rows[0]["step"], "->", rows[-1]["step"], "epoch", rows[-1]["epoch"])
    bad = []
    for i, r in enumerate(rows):
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                bad.append((i, k, v))
    print("nan_inf", len(bad))

    n = len(rows)
    for a, b, label in [
        (0, min(200, n), "first200"),
        (max(0, n // 2 - 100), min(n, n // 2 + 100), "mid200"),
        (max(0, n - 200), n, "last200"),
    ]:
        if b <= a:
            continue
        sub = rows[a:b]
        print(
            label,
            "steps",
            sub[0]["step"],
            "-",
            sub[-1]["step"],
            "total",
            f"{median(sub, 'total_loss'):.6g}",
            "heat",
            f"{median(sub, 'heat_loss'):.6g}",
            "peak_w",
            f"{median(sub, 'peak_weighted'):.6g}",
            "motion_w",
            f"{median(sub, 'motion_weighted'):.6g}",
            "valid",
            f"{median(sub, 'motion_valid_pair_ratio'):.3f}",
            "gtmot",
            f"{median(sub, 'mean_gt_motion_norm'):.3f}",
            "predmot",
            f"{median(sub, 'mean_pred_motion_norm'):.3f}",
            "resid",
            f"{median(sub, 'mean_motion_residual_norm'):.4f}",
            "fps",
            f"{median(sub, 'fps'):.2f}",
        )

    pts = sorted(d.glob("epoch*.pt"))
    print("checkpoints", [x.name for x in pts[-5:]], "latest", (d / "latest.pt").exists())
    log = d / "train.log"
    if log.exists():
        for line in log.read_text(errors="ignore").splitlines()[-5:]:
            print("LOG", line)

print("==== wait eval")
w = ROOT / "wait_eval.log"
print(w.read_text(errors="ignore")[-2000:] if w.exists() else "no wait log")
