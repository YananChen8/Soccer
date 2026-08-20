import csv
import math
from pathlib import Path

ROOT = Path("outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k5_k15_motion_lastpair_fast_20260701/fullft_cached_k5_last_motion_lastpair_fast_e5")
CSV = ROOT / "step_losses.csv"
LOG = ROOT / "train.log"

rows = []
with CSV.open(newline="") as f:
    for i, r in enumerate(csv.DictReader(f), start=2):
        parsed = {"line": i}
        for k, v in r.items():
            parsed[k] = v
        rows.append(parsed)

num_fields = [
    "total_loss",
    "heat_loss",
    "peak_raw",
    "motion_raw",
    "peak_weighted",
    "motion_weighted",
    "motion_valid_pair_ratio",
    "mean_gt_motion_norm",
    "mean_pred_motion_norm",
    "mean_motion_residual_norm",
    "fps",
]

def bad(v):
    try:
        x = float(v)
        return not math.isfinite(x)
    except Exception:
        return True

bad_rows = []
bad_by_field = {k: 0 for k in num_fields}
for r in rows:
    bad_fields = [k for k in num_fields if bad(r.get(k, ""))]
    if bad_fields:
        bad_rows.append((r["line"], int(float(r["step"])), bad_fields, r))
        for k in bad_fields:
            bad_by_field[k] += 1

print("rows", len(rows))
print("bad_row_count", len(bad_rows))
print("bad_by_field", bad_by_field)
if bad_rows:
    print("first_bad", bad_rows[0][0], "step", bad_rows[0][1], "fields", bad_rows[0][2])
    first_step = bad_rows[0][1]
    print("around_first_bad")
    for r in rows:
        step = int(float(r["step"]))
        if first_step - 8 <= step <= first_step + 12:
            print({k: r.get(k) for k in ["line", "epoch", "step", *num_fields]})
    print("bad_clusters")
    clusters = []
    current = []
    prev = None
    for _, step, fields, _ in bad_rows:
        if prev is None or step == prev + 1:
            current.append(step)
        else:
            clusters.append(current)
            current = [step]
        prev = step
    if current:
        clusters.append(current)
    print([(c[0], c[-1], len(c)) for c in clusters[:20]])

print("log_tail")
if LOG.exists():
    for line in LOG.read_text(errors="ignore").splitlines()[-40:]:
        print(line)
