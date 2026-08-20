import json
import time
from pathlib import Path

BASE = Path("/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate")
SUMMARY = BASE / "outputs/gsr/temporal_hrnet/temporal_calib_results_hub/lastpair_epoch_sweep_summary_20260702"

REPORTS = [
    BASE / "outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k5_k15_motion_lastpair_fast_20260701/report_eval_epoch_sweep_test_stride20",
    BASE / "outputs/gsr/temporal_hrnet/temporal_calib_results_hub/full_finetune_temporal_nbjw_k15_stage1_restart_20260701/report_eval_epoch_sweep_test_stride20",
]

EXPECTED = {
    "fullft_cached_k5_last_motion_lastpair_fast_e5": 5,
    "fullft_cached_k5_stage1_motion_lastpair_fast_e5": 5,
    "fullft_cached_k15_stage1_motion_lastpair_fast_e5_restart_stepckpt": 5,
}


def fmt(v, nd=4):
    if v is None:
        return "NA"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def collect():
    rows = []
    per_video = []
    seen_baseline = False
    for report in REPORTS:
        if not report.exists():
            continue
        for done in sorted(report.glob("*_epoch*/DONE.json")):
            meta = json.loads(done.read_text())
            batch_dir = report / meta["batch"]
            result_path = batch_dir / "test_results.json"
            if not result_path.exists():
                continue
            data = json.loads(result_path.read_text())
            for run, payload in data.items():
                if run == "baseline":
                    if seen_baseline:
                        continue
                    model, epoch = "baseline", 0
                    seen_baseline = True
                else:
                    if run != done.parent.name:
                        continue
                    model, epoch = meta["run"], int(meta["epoch"])
                agg = payload.get("aggregate", {})
                rows.append({"model": model, "epoch": epoch, **agg})
                for video, vals in payload.get("videos", {}).items():
                    per_video.append({"model": model, "epoch": epoch, "video": video, **vals})
    rows.sort(key=lambda r: (r["model"], r["epoch"]))
    per_video.sort(key=lambda r: (r["model"], r["epoch"], str(r["video"])))
    return rows, per_video


def write_tables():
    SUMMARY.mkdir(parents=True, exist_ok=True)
    rows, per_video = collect()
    standard_cols = ["JaC@5", "JaC@10", "JaC@15", "JaC@20", "MRE", "CR", "Final Score", "camera_smooth_l2_mean", "camera_smooth_l2_p95", "n_total"]
    legacy_cols = ["point_acc", "line_acc", "reproj_mean", "smooth_mean", "n_total"]

    def write_md(path, title, cols, source_rows):
        lines = [f"# {title}", "", "| model | epoch | " + " | ".join(cols) + " |", "|---|---:|" + "|".join(["---:" for _ in cols]) + "|"]
        for r in source_rows:
            nds = [2 if c in ("MRE", "reproj_mean") else 4 for c in cols]
            lines.append("| " + str(r["model"]) + " | " + str(r["epoch"]) + " | " + " | ".join(fmt(r.get(c), nd) for c, nd in zip(cols, nds)) + " |")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_per_video(path, title, cols):
        lines = [f"# {title}", "", "| model | epoch | video | " + " | ".join(cols) + " |", "|---|---:|---:|" + "|".join(["---:" for _ in cols]) + "|"]
        for r in per_video:
            nds = [2 if c in ("MRE", "reproj_mean") else 4 for c in cols]
            lines.append("| " + str(r["model"]) + " | " + str(r["epoch"]) + " | " + str(r["video"]) + " | " + " | ".join(fmt(r.get(c), nd) for c, nd in zip(cols, nds)) + " |")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_md(SUMMARY / "AGGREGATE_STANDARD_JAC_TABLE.md", "Aggregate Standard Metrics", standard_cols, rows)
    write_md(SUMMARY / "AGGREGATE_LEGACY_POINT_LINE_TABLE.md", "Aggregate Legacy Metrics", legacy_cols, rows)
    write_per_video(SUMMARY / "PER_VIDEO_STANDARD_JAC_TABLE.md", "Per-Video Standard Metrics", standard_cols)
    write_per_video(SUMMARY / "PER_VIDEO_LEGACY_POINT_LINE_TABLE.md", "Per-Video Legacy Metrics", legacy_cols)
    status = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "aggregate_rows": len(rows),
        "per_video_rows": len(per_video),
        "expected_epoch_rows": sum(EXPECTED.values()),
        "complete": len([r for r in rows if r["model"] != "baseline"]) >= sum(EXPECTED.values()),
    }
    (SUMMARY / "STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main():
    while True:
        status = write_tables()
        if status["complete"]:
            break
        time.sleep(600)


if __name__ == "__main__":
    main()
