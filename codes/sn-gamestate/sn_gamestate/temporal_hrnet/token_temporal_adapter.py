from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_temporal_adapter import field_adjacency


def heatmaps_to_tokens(heatmaps: torch.Tensor) -> torch.Tensor:
    """Convert [B,C,H,W] heatmaps to normalized (x,y,peak) tokens [B,C,3]."""
    b, c, h, w = heatmaps.shape
    flat = heatmaps.reshape(b, c, -1)
    peak, index = flat.max(dim=-1)
    x = (index % w).float() / max(w - 1, 1) * 2.0 - 1.0
    y = (index // w).float() / max(h - 1, 1) * 2.0 - 1.0
    return torch.stack((x, y, peak), dim=-1)


class _TemporalBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 1),
        )

    def forward(self, x):
        return F.relu(x + self.net(x), inplace=True)


class KeypointTokenTemporalAdapter(nn.Module):
    """Temporal adapter over sparse keypoint tokens, then warps current heatmaps."""

    def __init__(
        self,
        channels: int = 58,
        window_size: int = 50,
        architecture: str = "tcn",
        hidden: int = 64,
        residual_scale: float = 1.0,
        max_shift_px: float = 12.0,
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
            self.temporal = nn.Sequential(
                _TemporalBlock(hidden, 1),
                _TemporalBlock(hidden, 2),
                _TemporalBlock(hidden, 4),
                _TemporalBlock(hidden, 8),
                _TemporalBlock(hidden, 16),
            )
        elif architecture == "stgcn":
            self.temporal = nn.Sequential(
                _TemporalBlock(hidden, 1),
                _TemporalBlock(hidden, 2),
                _TemporalBlock(hidden, 4),
                _TemporalBlock(hidden, 8),
                _TemporalBlock(hidden, 16),
            )
            self.graph_self = nn.Linear(hidden, hidden)
            self.graph_neigh = nn.Linear(hidden, hidden)
            self.register_buffer("adjacency", field_adjacency(channels))
        elif architecture == "transformer":
            self.positional = nn.Parameter(torch.zeros(1, window_size, hidden))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
                dim_feedforward=hidden * 2,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        else:
            raise ValueError(f"unknown architecture: {architecture}")

        self.output = nn.Linear(hidden, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _encode(self, tokens):
        b, k, c, _ = tokens.shape
        x = self.input_proj(tokens)
        if self.architecture in {"tcn", "stgcn"}:
            x = x.permute(0, 2, 3, 1).reshape(b * c, self.hidden, k)
            x = self.temporal(x)[:, :, -1].reshape(b, c, self.hidden)
            if self.architecture == "stgcn":
                neighbours = torch.einsum("ij,bjh->bih", self.adjacency, x)
                x = F.relu(self.graph_self(x) + self.graph_neigh(neighbours), inplace=True)
            return x
        x = x.permute(0, 2, 1, 3).reshape(b * c, k, self.hidden)
        x = self.temporal(x + self.positional[:, :k])
        return x[:, -1].reshape(b, c, self.hidden)

    def forward(self, tokens: torch.Tensor, current_heatmap: torch.Tensor):
        """tokens [B,K,C,3], current_heatmap [B,C,H,W]."""
        b, c, h, w = current_heatmap.shape
        params = self.output(self._encode(tokens))
        shift = torch.tanh(params[..., :2])
        gate = torch.sigmoid(params[..., 2:3])

        theta = current_heatmap.new_zeros((b * c, 2, 3))
        theta[:, 0, 0] = 1
        theta[:, 1, 1] = 1
        theta[:, :, 2] = shift.reshape(b * c, 2) * current_heatmap.new_tensor(
            [2.0 * self.max_shift_px / max(w - 1, 1), 2.0 * self.max_shift_px / max(h - 1, 1)]
        )
        source = current_heatmap.reshape(b * c, 1, h, w)
        grid = F.affine_grid(theta, source.shape, align_corners=True)
        shifted = F.grid_sample(
            source, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).reshape(b, c, h, w)
        delta = gate.unsqueeze(-1) * (shifted - current_heatmap)
        return current_heatmap + self.residual_scale * delta, delta
