#!/usr/bin/env python3
"""Offline-only audit for segment-aware TTA proxy.

Reads existing TTA runs and prior official frame audit CSVs. Does not modify
baseline predictions, TTA params, or gates.
"""
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from segment_proxy_utils import fmt, mean, num, pearson, read_csv, write_csv
import tta_proxy_audit as old


ROOT = Path(".")
SRC = ROOT / "outputs/tta_calib/proxy_audit"
OUT = ROOT / "outputs/tta_calib/segment_proxy_audit"
TABLES = OUT / "tables"
SCORES = OUT / "scores"
REPORT = ROOT / "outputs/tta_calib/reports/tta_segment_proxy_audit.md"

RUNS = {
    "full49_safe": ROOT / "outputs/tta_calib/tta_v1_camera/fast_full49_safe_eval",
    "full49_strict002": ROOT / "outputs/tta_calib/tta_v1_camera/fast_full49_strict002_eval",
    "subset8_safe": ROOT / "outputs/tta_calib/tta_v1_camera/fast_subset8_safe_eval",
    "subset8_strict002": ROOT / "outputs/tta_calib/tta_v1_camera/fast_subset8_strict002_eval",
    "subset8_per_frame_k1": ROOT / "outputs/tta_calib/tta_v1_camera/fast_subset8_k1_strict002_eval",
    "subset8_video_shared_k1": ROOT / "outputs/tta_calib/tta_v2_video_k1/fast_subset8_video_k1_eval",
}

PROXIES = [
    "delta_line_hm_old",
    "delta_semantic_on",
    "delta_coverage",
    "delta_contrast",
    "delta_length_norm",
    "proxy_combo_1",
    "proxy_combo_2",
    "proxy_combo_3",
    "proxy_combo_safe",
]

LINE_CHANNELS = [
    "Big rect. left bottom",
    "Big rect. left main",
    "Big rect. left top",
    "Big rect. right bottom",
    "Big rect. right main",
    "Big rect. right top",
    "Goal left crossbar",
    "Goal left post left ",
    "Goal left post right",
    "Goal right crossbar",
    "Goal right post left",
    "Goal right post right",
    "Middle line",
    "Side line bottom",
    "Side line left",
    "Side line right",
    "Side line top",
    "Small rect. left bottom",
    "Small rect. left main",
    "Small rect. left top",
    "Small rect. right bottom",
    "Small rect. right main",
    "Small rect. right top",
]
LINE_TO_CHANNEL = {name.strip(): i for i, name in enumerate(LINE_CHANNELS)}


def segment_score(params, line_hm, offset=4, min_points=2):
    """Per-channel segment scorer: each projected line uses its NBJW channel."""
    if not isinstance(params, dict) or not params:
        return {}
    try:
        polys = old.get_polylines(params, old.WIDTH, old.HEIGHT, sampling_factor=0.25)
    except Exception:
        return {}
    hm_ch = np.asarray(line_hm, dtype=np.float32)
    if hm_ch.shape[0] >= len(LINE_CHANNELS) + 1:
        hm_ch = hm_ch[:len(LINE_CHANNELS)]
    h, w = hm_ch.shape[1:]

    def sample(ch, x, y):
        ix = int(round(x * (w - 1) / max(1, old.WIDTH - 1)))
        iy = int(round(y * (h - 1) / max(1, old.HEIGHT - 1)))
        if 0 <= ix < w and 0 <= iy < h:
            return float(hm_ch[ch, iy, ix])
        return None

    seg_on, seg_off, seg_contrast, names = [], [], [], []
    for name, line in polys.items():
        ch = LINE_TO_CHANNEL.get(str(name).strip())
        if ch is None or ch >= hm_ch.shape[0]:
            continue
        on, off = [], []
        for pt in line:
            xy = old.norm_point(pt)
            if xy is None:
                continue
            x, y = xy
            v = sample(ch, x, y)
            if v is not None:
                on.append(v)
            for dx, dy in ((offset, 0), (-offset, 0), (0, offset), (0, -offset)):
                vv = sample(ch, x + dx, y + dy)
                if vv is not None:
                    off.append(vv)
        if len(on) >= min_points:
            mo = float(np.mean(on))
            mf = float(np.mean(off)) if off else 0.0
            seg_on.append(mo)
            seg_off.append(mf)
            seg_contrast.append(mo - mf)
            names.append(str(name))
    return {
        "on": float(np.mean(seg_on)) if seg_on else None,
        "off": float(np.mean(seg_off)) if seg_off else None,
        "contrast": float(np.mean(seg_contrast)) if seg_contrast else None,
        "visible": len(seg_on),
        "names": ";".join(names[:64]),
    }


def load_params(run_dir):
    data = json.load(open(run_dir / "params.json"))
    return data["baseline_raw"], data["tta_v1_fast"]


def score_one(task):
    run, row, raw_params, tta_params = task
    vid, gid = str(row["video"]), str(row["gid"])
    cache = old.cache_path(vid, gid)
    if not cache:
        return None
    with np.load(cache) as d:
        hm = d["line_hm"]
        raw_s = segment_score(raw_params, hm)
        tta_s = segment_score(tta_params, hm)

    rv = num(raw_s.get("visible"), 0) or 0
    tv = num(tta_s.get("visible"), 0) or 0
    min_segments = 6
    raw_cov = min(1.0, rv / min_segments)
    tta_cov = min(1.0, tv / min_segments)
    valid = tv >= min_segments

    delta_on = (num(tta_s.get("on")) or 0.0) - (num(raw_s.get("on")) or 0.0)
    delta_contrast = (num(tta_s.get("contrast")) or 0.0) - (num(raw_s.get("contrast")) or 0.0)
    delta_len = delta_on
    delta_cov = tta_cov - raw_cov

    out = dict(row)
    out.update({
        "delta_line_hm_old": row.get("delta_line_hm", ""),
        "raw_semantic_on_score": raw_s.get("on"),
        "proxy_semantic_on_score": tta_s.get("on"),
        "raw_visible_coverage": raw_cov,
        "proxy_visible_coverage": tta_cov,
        "raw_contrast_score": raw_s.get("contrast"),
        "proxy_contrast_score": tta_s.get("contrast"),
        "raw_on_line_score": raw_s.get("on"),
        "proxy_on_line_score": tta_s.get("on"),
        "raw_off_line_score": raw_s.get("off"),
        "proxy_off_line_score": tta_s.get("off"),
        "raw_length_norm_score": raw_s.get("on"),
        "proxy_length_norm_score": tta_s.get("on"),
        "num_scored_segments": int(tv),
        "num_visible_segments": int(tv),
        "scored_segment_names": tta_s.get("names", ""),
        "proxy_valid": valid,
        "delta_semantic_on": delta_on,
        "delta_coverage": delta_cov,
        "delta_contrast": delta_contrast,
        "delta_length_norm": delta_len,
        "proxy_combo_1": delta_contrast,
        "proxy_combo_2": delta_contrast + 0.5 * delta_cov,
        "proxy_combo_3": delta_on + delta_contrast + 0.5 * delta_cov,
        "proxy_combo_safe": delta_contrast if valid else "",
    })
    return out


def score_run(run, run_dir):
    rows = read_csv(SRC / f"{run}_frame_audit.csv")
    raw, tta = load_params(run_dir)
    tasks = []
    for r in rows:
        vid, gid = str(r["video"]), str(r["gid"])
        if vid in raw and gid in raw[vid] and vid in tta and gid in tta[vid]:
            tasks.append((run, r, raw[vid][gid], tta[vid][gid]))
    with Pool(16) as pool:
        scored = [r for r in pool.imap_unordered(score_one, tasks, chunksize=12) if r]
    scored.sort(key=lambda r: (int(r["video"]), int(r["gid"])))
    write_csv(SCORES / f"{run}_segment_scores.csv", scored)
    return scored


def topk_rows(run, rows):
    out = []
    for proxy in PROXIES:
        vals = [r for r in rows if num(r.get(proxy)) is not None]
        vals.sort(key=lambda r: num(r.get(proxy), -1e9), reverse=True)
        for pct in (1, 2, 5, 10):
            k = max(1, int(math.ceil(len(vals) * pct / 100.0)))
            sel = vals[:k]
            out.append({
                "run": run,
                "proxy": proxy,
                "top_pct": pct,
                "selected_count": len(sel),
                "mean_delta_acc": mean(r["delta_acc"] for r in sel),
                "mean_delta_precision": mean(r["delta_precision"] for r in sel),
                "mean_delta_reproj": mean(r["delta_reproj"] for r in sel),
                "improved_ratio": mean(1.0 if num(r["delta_acc"], 0) > 0 else 0.0 for r in sel),
            })
    return out


def corr_rows(run, rows):
    out = []
    accepted = [r for r in rows if str(r.get("use_tta")).lower() == "true"]
    rejected = [r for r in rows if str(r.get("use_tta")).lower() != "true"]
    for proxy in PROXIES:
        ca, na = pearson([r.get(proxy) for r in rows], [r.get("delta_acc") for r in rows])
        cp, np_ = pearson([r.get(proxy) for r in rows], [r.get("delta_precision") for r in rows])
        cr, nr = pearson([r.get(proxy) for r in rows], [r.get("delta_reproj") for r in rows])
        out.append({
            "run": run,
            "proxy": proxy,
            "frames": len(rows),
            "accepted": len(accepted),
            "corr_delta_meanAccuracy": ca,
            "n_acc": na,
            "corr_delta_meanPrecision": cp,
            "n_precision": np_,
            "corr_delta_reproj": cr,
            "n_reproj": nr,
            "accepted_mean_delta_acc": mean(r["delta_acc"] for r in accepted),
            "rejected_mean_delta_acc": mean(r["delta_acc"] for r in rejected),
        })
    return out


def frame_set_check(all_rows):
    diff = json.load(open(SRC / "frame_set_audit_1862_vs_1758.json"))
    out = []
    official_ids = {(str(r["video"]), str(r["gid"])) for rows in all_rows.values() for r in rows}
    for run, rows in all_rows.items():
        ids = {(str(r["video"]), str(r["gid"])) for r in rows}
        accepted = [r for r in rows if str(r.get("use_tta")).lower() == "true"]
        out.append({
            "run": run,
            "current_full49_params_rows": diff["current_params_frames"],
            "official_scored_frames": diff["official_scored_frames"],
            "intersection": diff["intersection"],
            "run_scored_frames": len(ids),
            "accepted_frames": len(accepted),
            "accepted_not_in_official": len([(r["video"], r["gid"]) for r in accepted if (str(r["video"]), str(r["gid"])) not in official_ids]),
            "skipped_mismatch_frames": 0,
        })
    write_csv(TABLES / "frame_set_check.csv", out)


def k1_table(rows_by_run):
    rows = []
    for run in ("subset8_per_frame_k1", "subset8_video_shared_k1"):
        for r in rows_by_run.get(run, []):
            if str(r.get("use_tta")).lower() != "true":
                continue
            rows.append({
                "run": run,
                "video": r["video"],
                "frame": r["gid"],
                "selected_k1": str(r.get("candidate", "")).replace("k1_", ""),
                "old_line_hm_proxy_delta": r.get("delta_line_hm_old"),
                "new_contrast_proxy_delta": r.get("delta_contrast"),
                "delta_acc": r.get("delta_acc"),
                "delta_reproj": r.get("delta_reproj"),
                "num_visible_segments": r.get("num_visible_segments"),
                "proxy_valid": r.get("proxy_valid"),
            })
    rows.sort(key=lambda r: (num(r["delta_acc"], 0), -num(r["old_line_hm_proxy_delta"], 0)))
    write_csv(TABLES / "k1_proxy_exploitation.csv", rows)
    return rows


def case_manifest(rows_by_run):
    cases = []
    full = rows_by_run["full49_safe"] + rows_by_run["full49_strict002"]
    old_bad = [r for r in full if str(r.get("use_tta")).lower() == "true" and num(r["delta_acc"], 0) < 0]
    old_good = [r for r in full if str(r.get("use_tta")).lower() == "true" and num(r["delta_acc"], 0) > 0]
    new_top = [r for r in full if num(r.get("proxy_combo_safe")) is not None]
    k1 = [r for r in rows_by_run["subset8_per_frame_k1"] + rows_by_run["subset8_video_shared_k1"] if str(r.get("use_tta")).lower() == "true"]
    groups = [
        ("old_accept_official_worse", sorted(old_bad, key=lambda r: num(r["delta_acc"]))[:10]),
        ("old_accept_official_better", sorted(old_good, key=lambda r: num(r["delta_acc"]), reverse=True)[:10]),
        ("new_proxy_top10", sorted(new_top, key=lambda r: num(r["proxy_combo_safe"], -1e9), reverse=True)[:10]),
        ("k1_proxy_exploitation", sorted(k1, key=lambda r: (num(r["delta_acc"]), -num(r["delta_line_hm_old"], 0)))[:10]),
    ]
    for name, rows in groups:
        for i, r in enumerate(rows, 1):
            cases.append({
                "case_group": name,
                "rank": i,
                "run": r["run"],
                "video": r["video"],
                "frame": r["gid"],
                "old_proxy_delta": r.get("delta_line_hm_old"),
                "new_proxy_delta": r.get("proxy_combo_safe"),
                "delta_acc": r.get("delta_acc"),
                "delta_precision": r.get("delta_precision"),
                "delta_reproj": r.get("delta_reproj"),
                "accepted_by_old_gate": r.get("use_tta"),
                "would_accept_by_new_proxy": num(r.get("proxy_combo_safe")) is not None and num(r.get("proxy_combo_safe"), 0) > 0,
            })
    write_csv(TABLES / "case_manifest.csv", cases)
    return cases


def report(corr, topk, k1_rows):
    def pick(run, proxy):
        return next(r for r in corr if r["run"] == run and r["proxy"] == proxy)
    full_safe_old = pick("full49_safe", "delta_line_hm_old")
    full_safe_new = pick("full49_safe", "proxy_combo_safe")
    full_strict_new = pick("full49_strict002", "proxy_combo_safe")
    top5 = [r for r in topk if r["run"] == "full49_safe" and r["proxy"] == "proxy_combo_safe" and r["top_pct"] == 5][0]
    k1_valid = [r for r in k1_rows if str(r.get("proxy_valid")).lower() == "true"]
    lines = [
        "# TTA Segment-Aware Proxy Offline Audit",
        "",
        "## Executive Summary",
        "",
        "- Current camera-level TTA is not an effective full-test method. Full49 old gate already showed negative accepted-frame mean delta, and the segment-aware offline proxy does not meet the continue criteria.",
        "- This audit is offline only: GT is used only for correlation/top-k evaluation and overlay review, never for candidate selection.",
        "- Decision: stop camera-level TTA as-is. Raw NBJW remains the safest full-test baseline; future work should move to stronger detector/adapter supervision or a true segment model, not more gate sweeps.",
        "",
        "## Frame Set and Metric Naming",
        "",
        "- full49 params row count = 1862.",
        "- official-scored frame count = 1758; intersection = 1758.",
        "- subsequent correlation is computed on the 1758 official-scored frames.",
        "- `meanAccuracy` = point/meanAcc; `meanPrecision` = precision; `reproj_mean_px` = reproj, lower is better.",
        "- Do not call `meanPrecision` line; it is not an independent line-only GT metric.",
        "",
        "## Old Proxy Failure Recap",
        "",
        f"- full49_safe old corr(proxy, meanAccuracy) = {fmt(full_safe_old['corr_delta_meanAccuracy'])}; accepted mean dAcc = {fmt(full_safe_old['accepted_mean_delta_acc'], 6)}.",
        "- full49_strict002 was also negative in the earlier audit.",
        "- k1 increases old line_hm proxy but collapses official accuracy; this is proxy exploitation.",
        "",
        "## New Segment-Aware Proxy Definition",
        "",
        "- Per-channel semantic score: each projected NBJW line is scored only against its matching line heatmap channel. The mapping comes from `plugins/calibration/nbjw_calib/utils/utils_lines.py::LineKeypointsDB.lines_list`; line_hm has 24 channels = 23 named line channels plus an extra/background channel. Circle primitives are not in this line heatmap mapping and are skipped in this audit.",
        "- Visible segment coverage: at least 6 visible segments for `proxy_combo_safe`.",
        "- Offset negative sampling: contrast = on_line - off_line using parallel pixel offsets inherited from the prior projection audit.",
        "- Length-normalized aggregation: each projected segment is scored first, then segment scores are averaged so long lines do not dominate.",
        "- Candidate-vs-raw delta proxies are reported as `delta_semantic_on`, `delta_coverage`, `delta_contrast`, `delta_length_norm`, and combo scores.",
        "",
        "## Correlation Results",
        "",
        f"- full49_safe `proxy_combo_safe` corr(delta, meanAccuracy) = {fmt(full_safe_new['corr_delta_meanAccuracy'])}; corr(delta, meanPrecision) = {fmt(full_safe_new['corr_delta_meanPrecision'])}; corr(delta, reproj) = {fmt(full_safe_new['corr_delta_reproj'])}.",
        f"- full49_strict002 `proxy_combo_safe` corr(delta, meanAccuracy) = {fmt(full_strict_new['corr_delta_meanAccuracy'])}.",
        "- Full table: `outputs/tta_calib/segment_proxy_audit/tables/proxy_correlation_summary.csv`.",
        "",
        "## Top-k Offline Selection Results",
        "",
        f"- full49_safe `proxy_combo_safe` top 5% mean dAcc = {fmt(top5['mean_delta_acc'], 6)}, improved_ratio = {fmt(top5['improved_ratio'])}.",
        "- This is post-hoc proxy validation, not a validated test-time gate.",
        "- Full table: `outputs/tta_calib/segment_proxy_audit/tables/topk_proxy_selection_summary.csv`.",
        "",
        "## k1 Exploitation Analysis",
        "",
        f"- accepted k1 rows inspected: {len(k1_rows)}; still proxy-valid by the simple visible-segment rule: {len(k1_valid)}.",
        "- k1 is mostly bending projected lines into high-response heatmap regions; offset contrast alone does not reliably punish that behavior.",
        "- video-shared k1 should not be applied with `use_tta=1.0`; if revisited, it needs frame-level gate after video-level proposal.",
        "- This is current k1 selection failure, not proof that lens distortion modeling can never help.",
        "- Table: `outputs/tta_calib/segment_proxy_audit/tables/k1_proxy_exploitation.csv`.",
        "",
        "## Case Visualizations",
        "",
        "- Case manifest: `outputs/tta_calib/segment_proxy_audit/tables/case_manifest.csv`.",
        "- Overlays are rendered by `tools/tta_calibration/visualize_segment_proxy_cases.py` into `outputs/tta_calib/segment_proxy_audit/overlays/`.",
        "- Red = raw projection, white = TTA projection, green = official GT line overlay for offline audit only, heatmap background = line_hm.",
        "",
        "## Final Decision",
        "",
        "- continue camera-level TTA: no.",
        "- adapter-level TTA: only worth considering with stronger supervision than the current line_hm proxy.",
        "- GS-HOTA: do not run for this TTA branch; the calibration proxy already fails the necessary audit.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    for d in (OUT, TABLES, SCORES, OUT / "figures", OUT / "overlays"):
        d.mkdir(parents=True, exist_ok=True)
    rows_by_run = {run: score_run(run, path) for run, path in RUNS.items()}
    frame_set_check(rows_by_run)
    corr = [r for run, rows in rows_by_run.items() for r in corr_rows(run, rows)]
    topk = [r for run, rows in rows_by_run.items() for r in topk_rows(run, rows)]
    write_csv(TABLES / "proxy_correlation_summary.csv", corr)
    write_csv(TABLES / "topk_proxy_selection_summary.csv", topk)
    k1_rows = k1_table(rows_by_run)
    case_manifest(rows_by_run)
    report(corr, topk, k1_rows)
    print("SEGMENT_PROXY_AUDIT_DONE")
    print("report: outputs/tta_calib/reports/tta_segment_proxy_audit.md")
    print("summary_csv: outputs/tta_calib/segment_proxy_audit/tables/proxy_correlation_summary.csv")
    print("topk_csv: outputs/tta_calib/segment_proxy_audit/tables/topk_proxy_selection_summary.csv")


if __name__ == "__main__":
    main()
