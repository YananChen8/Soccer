import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

try:
    from train_heatmap_embedding import HRLiteEmbedding
except ModuleNotFoundError:
    from tmp_train_heatmap_embedding import HRLiteEmbedding


KS = [5, 10, 15, 20, 40, 80]


def load_manifest(root, split):
    path = Path(root) / split / "manifest.csv"
    with path.open() as f:
        return list(csv.DictReader(f))


def get_heatmap(root, row, cache):
    path = Path(root) / row["path"]
    if path.suffix == ".npy":
        key = str(path)
        if key not in cache:
            cache.clear()
            cache[key] = np.load(path, mmap_mode="r")
        return np.asarray(cache[key][int(row["offset"])])
    return np.load(path)["heatmap"]


def load_features(root, rows, max_rows=0, model=None, device="cpu", batch_size=64):
    feats = []
    kept = rows[:max_rows] if max_rows else rows
    cache = {}
    if model is None:
        for r in kept:
            hm = get_heatmap(root, r, cache).astype(np.float32)
            # ponytail: global average per channel is the cheap baseline; CNN embedding replaces this.
            f = hm.mean(axis=(1, 2))
            f -= f.mean()
            n = np.linalg.norm(f)
            feats.append(f / n if n > 1e-8 else f)
    else:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(kept), batch_size):
                xs = [get_heatmap(root, r, cache).astype(np.float32) for r in kept[start:start + batch_size]]
                x = torch.from_numpy(np.stack(xs)).to(device)
                feats.append(model(x).cpu().numpy())
        feats = [x for arr in feats for x in arr]
    return kept, np.stack(feats).astype(np.float32)


def load_model(path, device):
    if not path:
        return None
    ck = torch.load(path, map_location=device)
    model = HRLiteEmbedding()
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    return model


def evaluate(root, split, db_split, max_queries, max_db, checkpoint, device, batch_size):
    model = load_model(checkpoint, device)
    q_rows, q = load_features(root, load_manifest(root, split), max_queries, model, device, batch_size)
    d_rows, db = load_features(root, load_manifest(root, db_split), max_db, model, device, batch_size)
    sims = q @ db.T
    results = {}
    details = []
    for k in KS:
        vals = []
        hit_same_video = 0
        for i, qr in enumerate(q_rows):
            score = sims[i].copy()
            for j, dr in enumerate(d_rows):
                if qr["video"] == dr["video"] and abs(int(qr["frame"]) - int(dr["frame"])) <= k:
                    score[j] = -np.inf
            best = int(np.argmax(score))
            br = d_rows[best]
            same = qr["video"] == br["video"]
            if same:
                hit_same_video += 1
                gap_minus_k = max(0, abs(int(qr["frame"]) - int(br["frame"])) - k)
            else:
                gap_minus_k = None
            vals.append(gap_minus_k)
            if k == KS[0]:
                details.append({
                    "query_video": qr["video"],
                    "query_frame": qr["frame"],
                    "match_video": br["video"],
                    "match_frame": br["frame"],
                    "same_video": same,
                    "score": float(score[best]),
                    "gap_minus_k": gap_minus_k,
                })
        finite = [v for v in vals if v is not None]
        results[str(k)] = {
            "queries": len(q_rows),
            "db": len(d_rows),
            "same_video_rate": hit_same_video / max(1, len(q_rows)),
            "mean_gap_minus_k": float(np.mean(finite)) if finite else None,
            "median_gap_minus_k": float(np.median(finite)) if finite else None,
        }
    return results, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--db-split", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--max-db", type=int, default=0)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    db_split = args.db_split or args.split
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results, details = evaluate(args.cache_root, args.split, db_split, args.max_queries, args.max_db, args.checkpoint, torch.device(args.device), args.batch_size)
    (out / "retrieval_metrics.json").write_text(json.dumps(results, indent=2))
    with (out / "retrieval_details_k5.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_video", "query_frame", "match_video", "match_frame", "same_video", "score", "gap_minus_k"])
        w.writeheader()
        w.writerows(details)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
