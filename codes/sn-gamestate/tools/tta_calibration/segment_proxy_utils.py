#!/usr/bin/env python3
"""Small helpers for offline segment-proxy audit."""
import csv
import math
from pathlib import Path

import numpy as np


def num(x, default=None):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def pearson(xs, ys):
    vals = [(num(x), num(y)) for x, y in zip(xs, ys)]
    vals = [(x, y) for x, y in vals if x is not None and y is not None]
    if len(vals) < 3:
        return None, len(vals)
    x = np.array([v[0] for v in vals], dtype=float)
    y = np.array([v[1] for v in vals], dtype=float)
    if x.std() == 0 or y.std() == 0:
        return None, len(vals)
    return float(np.corrcoef(x, y)[0, 1]), len(vals)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not fieldnames and rows:
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames or [])
        w.writeheader()
        w.writerows(rows)


def mean(vals):
    vals = [num(v) for v in vals]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def fmt(v, nd=4):
    return "NA" if v is None else f"{v:.{nd}f}"

