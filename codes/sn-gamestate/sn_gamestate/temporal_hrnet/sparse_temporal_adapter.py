from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


FIELD_LINES = [
    [24, 25], [5, 31, 46, 34, 25], [4, 5],
    [26, 27], [6, 33, 56, 36, 26], [6, 7],
    [32, 48, 38, 50, 42, 53, 35, 54, 43, 52, 39, 49],
    [31, 37, 47, 41, 34], [33, 40, 55, 44, 36],
    [16, 12], [16, 17], [12, 13],
    [15, 19], [15, 14], [19, 18],
    [2, 32, 51, 35, 29],
    [28, 29, 30], [1, 4, 8, 13, 17, 20, 24, 28],
    [3, 7, 11, 14, 18, 23, 27, 30], [1, 2, 3],
    [20, 21], [9, 21], [8, 9],
    [22, 23], [10, 22], [10, 11],
]


def field_adjacency(channels: int) -> torch.Tensor:
    adjacency = torch.eye(channels)
    for line in FIELD_LINES:
        ids = [keypoint_id - 1 for keypoint_id in line if keypoint_id <= channels]
        for left, right in zip(ids, ids[1:]):
            adjacency[left, right] = 1
            adjacency[right, left] = 1
    degree = adjacency.sum(dim=1).clamp_min(1)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


class _TemporalTCN(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 3, padding=8, dilation=8),
            nn.ReLU(inplace=True),
        )

    def forward(self, tokens):
        batch, steps, nodes, hidden = tokens.shape
        x = tokens.permute(0, 2, 3, 1).reshape(batch * nodes, hidden, steps)
        x = self.net(x)[:, :, -1]
        return x.reshape(batch, nodes, hidden)


class _TemporalTransformer(nn.Module):
    def __init__(self, hidden: int, heads: int, layers: int, max_window: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.position = nn.Parameter(torch.zeros(1, max_window, hidden))

    def forward(self, tokens):
        batch, steps, nodes, hidden = tokens.shape
        x = tokens.permute(0, 2, 1, 3).reshape(batch * nodes, steps, hidden)
        x = self.encoder(x + self.position[:, :steps])
        return x[:, -1].reshape(batch, nodes, hidden)


class _STGCN(nn.Module):
    def __init__(self, hidden: int, adjacency: torch.Tensor):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.temporal = nn.Sequential(
            nn.Conv2d(hidden, hidden, (3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, (3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
        )
        self.graph = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )

    def forward(self, tokens):
        x = tokens.permute(0, 3, 1, 2)
        x = self.temporal(x)[:, :, -1].transpose(1, 2)
        x = torch.einsum("ij,bjh->bih", self.adjacency, x)
        return self.graph(x)


class SparseTemporalKeypointAdapter(nn.Module):
    """Temporal model over per-keypoint peak tokens, followed by heatmap warping."""

    def __init__(
        self,
        channels: int,
        window_size: int,
        architecture: str = "tcn",
        hidden: int = 64,
        residual_scale: float = 1.0,
        max_shift_px: float = 8.0,
    ):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.architecture = architecture
        self.hidden = hidden
        self.residual_scale = residual_scale
        self.max_shift_px = max_shift_px
        self.input_proj = nn.Linear(3, hidden)
        if architecture == "tcn":
            self.temporal = _TemporalTCN(hidden)
        elif architecture == "stgcn":
            self.temporal = _STGCN(hidden, field_adjacency(channels))
        elif architecture == "transformer":
            self.temporal = _TemporalTransformer(hidden, heads=4, layers=2, max_window=window_size)
        else:
            raise ValueError(f"unknown sparse architecture: {architecture}")
        self.offset_head = nn.Linear(hidden, 3)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    @staticmethod
    def _tokens(heatmaps):
        batch, steps, channels, height, width = heatmaps.shape
        flat = heatmaps.reshape(batch, steps, channels, -1)
        confidence, index = flat.max(dim=-1)
        x = (index.remainder(width).float() / max(width - 1, 1)) * 2 - 1
        y = (index.div(width, rounding_mode="floor").float() / max(height - 1, 1)) * 2 - 1
        return torch.stack((x, y, confidence), dim=-1)

    def _warp(self, current, offsets):
        batch, channels, height, width = current.shape
        dx = torch.tanh(offsets[..., 0]) * self.max_shift_px
        dy = torch.tanh(offsets[..., 1]) * self.max_shift_px
        gain = 1.0 + 0.25 * torch.tanh(offsets[..., 2])
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=current.device, dtype=current.dtype),
            torch.linspace(-1, 1, width, device=current.device, dtype=current.dtype),
            indexing="ij",
        )
        grid = torch.stack((xx, yy), dim=-1)[None, None].expand(batch, channels, -1, -1, -1).clone()
        grid[..., 0] -= (2 * dx / max(width - 1, 1))[:, :, None, None]
        grid[..., 1] -= (2 * dy / max(height - 1, 1))[:, :, None, None]
        warped = F.grid_sample(
            current.reshape(batch * channels, 1, height, width),
            grid.reshape(batch * channels, height, width, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(batch, channels, height, width)
        return warped * gain[:, :, None, None]

    def forward(self, heatmap_sequence):
        current = heatmap_sequence[:, -1]
        if self.residual_scale == 0:
            return current, torch.zeros_like(current)
        tokens = self.input_proj(self._tokens(heatmap_sequence))
        offsets = self.offset_head(self.temporal(tokens))
        if not self.training and offsets.detach().abs().max().item() == 0:
            return current, torch.zeros_like(current)
        warped = self._warp(current, offsets)
        delta = warped - current
        return current + self.residual_scale * delta, delta
