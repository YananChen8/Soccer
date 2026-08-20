import argparse
import csv
import re
from pathlib import Path


PAT = re.compile(
    r"epoch=(?P<epoch>\d+).*?step=(?P<step>\d+).*?"
    r"loss=(?P<loss>[\deE+\-.]+).*?heat=(?P<heat>[\deE+\-.]+).*?"
    r"peak=(?P<peak>[\deE+\-.]+).*?motion=(?P<motion>[\deE+\-.]+).*?fg=(?P<fg>[\deE+\-.]+)"
)


def parse_log(path):
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            m = PAT.search(line)
            if not m:
                continue
            item = {k: float(v) for k, v in m.groupdict().items()}
            item["epoch"] = int(item["epoch"])
            item["step"] = int(item["step"])
            rows.append(item)
    return rows


def fmt(v, nd=6):
    return "NA" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = []
    for log in sorted(root.glob("fullft_*_k3/logs/train.log")):
        run = log.parts[-3]
        rows = parse_log(log)
        for r in rows:
            all_rows.append({"run": run, **r})
        if rows:
            first, last = rows[0], rows[-1]
            summary.append({
                "run": run,
                "n_points": len(rows),
                "first_step": first["step"],
                "last_step": last["step"],
                "first_loss": first["loss"],
                "last_loss": last["loss"],
                "first_heat": first["heat"],
                "last_heat": last["heat"],
                "first_peak": first["peak"],
                "last_peak": last["peak"],
                "first_motion": first["motion"],
                "last_motion": last["motion"],
                "first_fg": first["fg"],
                "last_fg": last["fg"],
            })
        else:
            summary.append({"run": run, "n_points": 0})
    if all_rows:
        with (out / "training_curves_lossfix.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    with (out / "training_curves_lossfix_summary.csv").open("w", newline="") as f:
        fields = sorted({k for row in summary for k in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# Lossfix Training Curve Summary", "", "| run | points | step first-last | loss first-last | heat first-last | peak first-last | motion first-last | fg first-last |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        if not r.get("n_points"):
            lines.append(f"| {r['run']} | 0 | NA | NA | NA | NA | NA | NA |")
            continue
        lines.append(
            f"| {r['run']} | {r['n_points']} | {r['first_step']}-{r['last_step']} | "
            f"{fmt(r['first_loss'])}-{fmt(r['last_loss'])} | {fmt(r['first_heat'])}-{fmt(r['last_heat'])} | "
            f"{fmt(r['first_peak'])}-{fmt(r['last_peak'])} | {fmt(r['first_motion'])}-{fmt(r['last_motion'])} | "
            f"{fmt(r['first_fg'])}-{fmt(r['last_fg'])} |"
        )
    (out / "training_curves_lossfix_summary.md").write_text("\n".join(lines) + "\n")
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
