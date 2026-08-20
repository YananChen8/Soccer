import csv
import json
from pathlib import Path

root = Path("outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k3_official_aux_20260701")
out = root / "report_eval_visual_20260701"
out.mkdir(exist_ok=True)

rows = []
for p in sorted(root.glob("fullft_offaux_*/eval_test116_123_stride20/results.json")):
    run = p.parts[-3]
    data = json.load(open(p))
    for vid, item in sorted(data["videos"].items(), key=lambda kv: int(kv[0])):
        for row_name, key in [("baseline", "baseline"), (run, "feature_fusion")]:
            m = item[key]
            rows.append({
                "run": row_name,
                "source_run": run,
                "video": vid,
                "point_acc": m.get("point_acc"),
                "line_acc": m.get("line_acc"),
                "reproj_mean": m.get("reproj_mean"),
                "smooth_mean": m.get("smooth_mean"),
                "n_total": m.get("n_total"),
                "n_scored": m.get("n_scored"),
            })

with (out / "test116_123_per_video_metrics_long.csv").open("w", newline="") as f:
    fields = list(rows[0])
    writer = csv.DictWriter(f, fields)
    writer.writeheader()
    writer.writerows(rows)

def avg(vals, key):
    vals = [float(v[key]) for v in vals if v.get(key) is not None]
    return sum(vals) / len(vals) if vals else None

def fmt(v, nd=4):
    return "NA" if v is None else f"{v:.{nd}f}"

lines = [
    "# Official-Aux Test116-123 Per-Video Metrics",
    "",
    "| source_run | row | video | point | line | reproj | smooth | frames |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['source_run']} | {r['run']} | {r['video']} | "
        f"{fmt(r['point_acc'])} | {fmt(r['line_acc'])} | {fmt(r['reproj_mean'], 2)} | "
        f"{fmt(r['smooth_mean'], 1)} | {r['n_total']} |"
    )

lines += [
    "",
    "# Aggregate",
    "",
    "| run | point | line | reproj | smooth | frames |",
    "|---|---:|---:|---:|---:|---:|",
]

source_runs = sorted({r["source_run"] for r in rows})
aggregate_order = ["baseline"] + sorted({r["run"] for r in rows if r["run"] != "baseline"})
for name in aggregate_order:
    vals = [r for r in rows if r["run"] == name]
    if name == "baseline":
        vals = [r for r in vals if r["source_run"] == source_runs[0]]
    lines.append(
        f"| {name} | {fmt(avg(vals, 'point_acc'))} | {fmt(avg(vals, 'line_acc'))} | "
        f"{fmt(avg(vals, 'reproj_mean'), 2)} | {fmt(avg(vals, 'smooth_mean'), 1)} | "
        f"{sum(int(v['n_total']) for v in vals)} |"
    )

(out / "test116_123_per_video_metrics.md").write_text("\n".join(lines) + "\n")
print(out / "test116_123_per_video_metrics.md")
