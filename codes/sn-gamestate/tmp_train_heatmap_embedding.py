import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class CacheDataset(Dataset):
    def __init__(self, root, split, pos_window):
        self.root = Path(root)
        with (self.root / split / "manifest.csv").open() as f:
            self.rows = list(csv.DictReader(f))
        self.by_video = {}
        for i, r in enumerate(self.rows):
            self.by_video.setdefault(r["video"], []).append(i)
        self.pos_window = pos_window
        self._cache = {}

    def __len__(self):
        return len(self.rows)

    def load(self, idx):
        r = self.rows[idx]
        path = self.root / r["path"]
        if path.suffix == ".npy":
            key = str(path)
            if key not in self._cache:
                self._cache.clear()
                self._cache[key] = np.load(path, mmap_mode="r")
            x = np.asarray(self._cache[key][int(r["offset"])]).astype(np.float32)
        else:
            x = np.load(path)["heatmap"].astype(np.float32)
        return torch.from_numpy(x), r

    def __getitem__(self, idx):
        x, r = self.load(idx)
        frame = int(r["frame"])
        same = self.by_video[r["video"]]
        pos = [j for j in same if j != idx and abs(int(self.rows[j]["frame"]) - frame) <= self.pos_window]
        pos_idx = random.choice(pos) if pos else idx
        xp, _ = self.load(pos_idx)
        return x, xp


class HRLiteEmbedding(nn.Module):
    def __init__(self, in_ch=82, dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.high = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.low = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(96, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(224, 160, 1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(160, dim),
        )

    def forward(self, x):
        s = self.stem(x)
        h = self.high(s)
        l = F.interpolate(self.low(s), size=h.shape[-2:], mode="bilinear", align_corners=False)
        z = self.head(torch.cat([h, l], dim=1))
        return F.normalize(z, dim=1)


def info_nce(z1, z2, temp):
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.T / temp
    sim.fill_diagonal_(-1e9)
    n = z1.shape[0]
    labels = torch.arange(2 * n, device=z.device)
    labels = (labels + n) % (2 * n)
    return F.cross_entropy(sim, labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-window", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.07)
    args = ap.parse_args()

    device = torch.device(args.device)
    ds = CacheDataset(args.cache_root, args.split, args.pos_window)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    model = HRLiteEmbedding().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    it = iter(dl)
    for step in range(1, args.steps + 1):
        try:
            x1, x2 = next(it)
        except StopIteration:
            it = iter(dl)
            x1, x2 = next(it)
        x1, x2 = x1.to(device), x2.to(device)
        loss = info_nce(model(x1), model(x2), args.temperature)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == 1:
            print(f"step={step} loss={float(loss):.4f}", flush=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
