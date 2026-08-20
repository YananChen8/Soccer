"""Archive train12 cached-supervision metrics for token temporal adapters."""
import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from sn_gamestate.temporal_hrnet import KeypointTokenTemporalAdapter, heatmaps_to_tokens


SNG = "/remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate"
CKPT = f"{SNG}/outputs/gsr/temporal_hrnet/quick_subset12"
ADAPTERS = {
    "baseline": None,
    "old_tcn_k50_ms12_ep1": f"{CKPT}/tcn_k50/kp_adapter_tcn_k50.pt",
    "old_stgcn_k50_ms12_ep1": f"{CKPT}/stgcn_k50/kp_adapter_stgcn_k50.pt",
    "old_transformer_k50_ms12_ep1": f"{CKPT}/transformer_k50/kp_adapter_transformer_k50.pt",
}


def fg_mse(pred, gt, mask):
    fg = (gt > 0.1).float() * mask[:, :, None, None]
    return (((pred - gt) ** 2 * fg).sum() / (fg.sum() + 1e-6)).item()


def load_adapter(path, device):
    if path is None:
        return None, 1, {}
    ck = torch.load(path, map_location=device)
    if ck.get("which") != "kp_token":
        raise ValueError(f"not a token checkpoint: {path}")
    model = KeypointTokenTemporalAdapter(
        channels=ck["channels"],
        window_size=ck["window_size"],
        architecture=ck["architecture"],
        hidden=ck.get("hidden", 64),
        residual_scale=ck.get("residual_scale", 1.0),
        max_shift_px=ck.get("max_shift_px", 12.0),
    )
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck["window_size"], {k: ck.get(k) for k in ["architecture", "window_size", "residual_scale", "max_shift_px", "steps"]}


def pad_tokens(buf, k):
    vals = list(buf)
    if len(vals) < k:
        vals = [vals[0]] * (k - len(vals)) + vals
    return torch.stack(vals, dim=0).unsqueeze(0)


def peak_xy(heatmap):
    c, h, w = heatmap.shape
    flat = heatmap.reshape(c, -1)
    idx = flat.argmax(dim=1)
    x = (idx % w).float()
    y = (idx // w).float()
    return torch.stack([x, y], dim=1)


def eval_one(cache_root, split, name, path, device):
    model, k, meta = load_adapter(path, device)
    totals = {
        "fg_base": 0.0,
        "fg_ref": 0.0,
        "shift_px": 0.0,
        "peak_step_base": 0.0,
        "peak_step_ref": 0.0,
        "frames": 0,
        "videos": 0,
    }
    for vdir in sorted((Path(cache_root) / split).iterdir()):
        if not vdir.is_dir():
            continue
        files = sorted(vdir.glob("frame_*.npz"))
        if not files:
            continue
        totals["videos"] += 1
        token_buf = deque(maxlen=k)
        prev_base, prev_ref = None, None
        for file in files:
            d = np.load(file)
            current = torch.from_numpy(d["kp_hm"].astype(np.float32)).unsqueeze(0).to(device)
            gt = torch.from_numpy(d["kp_gt"].astype(np.float32)).unsqueeze(0).to(device)
            mask = torch.from_numpy(d["kp_mask"].astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                token_buf.append(heatmaps_to_tokens(current)[0].detach())
                if model is None:
                    refined = current
                else:
                    refined, _delta = model(pad_tokens(token_buf, k).to(device), current)
                totals["fg_base"] += fg_mse(current.float(), gt, mask)
                totals["fg_ref"] += fg_mse(refined.float(), gt, mask)
                base_xy = peak_xy(current[0]).detach().cpu()
                ref_xy = peak_xy(refined[0]).detach().cpu()
                present = mask[0].detach().cpu() > 0
                if present.any():
                    totals["shift_px"] += torch.norm(ref_xy[present] - base_xy[present], dim=1).mean().item()
                    if prev_base is not None:
                        totals["peak_step_base"] += torch.norm(base_xy[present] - prev_base[present], dim=1).mean().item()
                        totals["peak_step_ref"] += torch.norm(ref_xy[present] - prev_ref[present], dim=1).mean().item()
                prev_base, prev_ref = base_xy, ref_xy
            totals["frames"] += 1
    denom = max(totals["frames"], 1)
    step_denom = max(totals["frames"] - totals["videos"], 1)
    return {
        "meta": meta,
        "videos": totals["videos"],
        "frames": totals["frames"],
        "fg_mse_base": totals["fg_base"] / denom,
        "fg_mse_refined": totals["fg_ref"] / denom,
        "fg_mse_delta": (totals["fg_ref"] - totals["fg_base"]) / denom,
        "mean_shift_px": totals["shift_px"] / denom,
        "peak_step_base": totals["peak_step_base"] / step_denom,
        "peak_step_refined": totals["peak_step_ref"] / step_denom,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--split", default="train12")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    results = {}
    for name, path in ADAPTERS.items():
        if (out_dir / "train12_old_models_metrics.json").exists():
            results = json.loads((out_dir / "train12_old_models_metrics.json").read_text(encoding="utf-8"))
        if name in results:
            print(f"SKIP {name}", flush=True)
            continue
        print(f"START {name}", flush=True)
        results[name] = eval_one(args.cache_root, args.split, name, path, device)
        (out_dir / "train12_old_models_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        write_markdown(out_dir, results)
        print(f"DONE {name} {results[name]}", flush=True)
    (out_dir / "train12_old_models_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(out_dir, results)
    print((out_dir / "TRAIN12_OLD_MODELS.md").read_text(encoding="utf-8"))


def write_markdown(out_dir, results):
    lines = [
        "| model | frames | fg_base | fg_refined | delta | shift_px | step_base | step_refined |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['frames']} | {r['fg_mse_base']:.6f} | {r['fg_mse_refined']:.6f} | "
            f"{r['fg_mse_delta']:.6f} | {r['mean_shift_px']:.3f} | "
            f"{r['peak_step_base']:.3f} | {r['peak_step_refined']:.3f} |"
        )
    (out_dir / "TRAIN12_OLD_MODELS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((out_dir / "TRAIN12_OLD_MODELS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
