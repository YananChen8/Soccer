import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-json", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.sweep_json).read_text())
    rows = [r for r in data.get("rows", []) if r.get("ok")]
    best = data.get("best") or (max(rows, key=lambda r: r["frames_per_sec"]) if rows else None)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Cached Training Speed Recommendation", ""]
    if not best:
        lines += ["No successful sweep configuration."]
    else:
        lines += [
            "## Best",
            "",
            f"- batch_size: `{best['batch_size']}`",
            f"- num_workers: `{best['num_workers']}`",
            f"- frames_per_sec: `{best['frames_per_sec']:.3f}`",
            f"- seconds: `{best['seconds']:.1f}` for `{best['frames']}` frames / `{best['steps']}` steps",
            f"- last loss: `{best.get('loss')}`",
            f"- last heat: `{best.get('heat')}`",
            f"- last peak_term: `{best.get('peak_term')}`",
            "",
            "## Recommended Next Train Settings",
            "",
            f"- Use cached train tensors.",
            f"- batch_size=`{best['batch_size']}`.",
            f"- num_workers=`{best['num_workers']}`.",
            "- Keep HRNet LR=3e-6, adapter LR=3e-5, grad_clip_norm=1.0.",
            "- Keep official masked MSE dominant and peak/motion terms scaled to about 0.2x heat.",
            "- Use fast heatmap proxy eval first; run full calibration eval only for candidates that do not regress peak distance.",
        ]
    lines += ["", "## All Sweep Rows", "", "| batch | workers | ok | frames/s | seconds | loss | heat | peak_term | error |", "|---:|---:|---|---:|---:|---:|---:|---:|---|"]
    for r in data.get("rows", []):
        lines.append(
            f"| {r.get('batch_size')} | {r.get('num_workers')} | {r.get('ok')} | "
            f"{fmt(r.get('frames_per_sec'))} | {fmt(r.get('seconds'),1)} | "
            f"{fmt(r.get('loss'))} | {fmt(r.get('heat'))} | {fmt(r.get('peak_term'))} | "
            f"{str(r.get('error',''))[:80]} |"
        )
    (out / "RECOMMENDED_SPEED_CONFIG.md").write_text("\n".join(lines) + "\n")
    (out / "recommended_speed_config.json").write_text(json.dumps({"best": best, "rows": data.get("rows", [])}, indent=2))
    print(json.dumps({"best": best, "out_dir": str(out)}, indent=2), flush=True)


def fmt(v, nd=4):
    if v is None:
        return "NA"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


if __name__ == "__main__":
    main()
