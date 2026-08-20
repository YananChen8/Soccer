"""Merge per-GPU full-test JSONs -> per-adapter aggregate (frame-weighted)."""
import glob
import json
import sys

OUT = "/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate/outputs/gsr/temporal_hrnet/analysis/full_test"
ADAPTERS = ["baseline", "token_tcn_k50", "token_stgcn_k50", "token_transformer_k50"]

videos = {}
for f in glob.glob(f"{OUT}/g*.json"):
    videos.update(json.load(open(f)))

print(f"videos merged: {len(videos)}")
agg = {}
for a in ADAPTERS:
    pt_sum = ln_sum = rj_sum = w = 0.0
    nscored = ntotal = 0
    ms_sum = ms_w = 0.0
    nvid = 0
    for vid, res in videos.items():
        if a not in res or res[a]["point_acc"] is None:
            continue
        n = res[a]["n_scored"]
        pt_sum += res[a]["point_acc"] * n
        ln_sum += res[a]["line_acc"] * n
        if res[a]["reproj_mean"] is not None:
            rj_sum += res[a]["reproj_mean"] * n
        w += n
        nscored += n; ntotal += res[a]["n_total"]
        ms_sum += res[a]["adapter_ms_per_frame"] * res[a]["n_total"]; ms_w += res[a]["n_total"]
        nvid += 1
    agg[a] = {
        "point_acc": pt_sum / w if w else None,
        "line_acc": ln_sum / w if w else None,
        "reproj_mean": rj_sum / w if w else None,
        "completeness": nscored / ntotal if ntotal else None,
        "adapter_ms_per_frame": ms_sum / ms_w if ms_w else 0.0,
        "n_videos": nvid, "n_frames_scored": nscored,
    }

json.dump({"per_adapter": agg, "per_video": videos}, open(f"{OUT}/merged.json", "w"), indent=2)
print(f"{'adapter':24s} {'point_acc':>10s} {'line_acc':>10s} {'reproj_mean':>12s} {'compl':>7s} {'adapter_ms':>11s}")
base = agg["baseline"]["point_acc"]
for a in ADAPTERS:
    r = agg[a]
    d = "" if a == "baseline" else f"  (Δpt {r['point_acc']-base:+.4f})"
    print(f"{a:24s} {r['point_acc']:>10.4f} {r['line_acc']:>10.4f} "
          f"{r['reproj_mean']:>12.2f} {r['completeness']:>7.3f} {r['adapter_ms_per_frame']:>10.2f}ms{d}")
print(f"\nmerged -> {OUT}/merged.json")
