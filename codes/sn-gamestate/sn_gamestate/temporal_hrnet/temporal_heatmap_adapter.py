"""Temporal Heatmap Adapter (Plan A) for NBJW/PnLCalib HRNet.

Inserts AFTER the HRNet output heatmaps, BEFORE decode. Refines the current
frame's heatmap using a short window of recent heatmaps:

    H_seq [B, K, C, h, w]  -->  H_refined [B, C, h, w],  delta [B, C, h, w]
    H_refined = H_t + residual_scale * delta

Design contract (do not break):
  * Last conv is zero-initialised -> delta == 0 at init -> H_refined == H_t.
    So loading the original HRNet checkpoint and running with an untrained
    adapter reproduces the original output exactly.
  * residual_scale == 0  -> hard passthrough (H_refined is H_t, no graph cost).
  * enabled flag lives in the caller; this module is pure nn.Module.

ponytail: depthwise Conv3D over time + 1x1 channel-mix + 3x3 decoder. No point-id
/ primitive embeddings in v1 (spec). convgru is a small real cell, not a raise,
so config adapter_type=convgru_stub doesn't crash.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def pad_window(h_seq: torch.Tensor, k: int) -> torch.Tensor:
    """Left-pad a [B, t, C, h, w] window to length k by repeating the oldest
    frame. t may be < k at video start / after a shot boundary. Never crashes.
    Returns [B, k, C, h, w]. If t > k, keeps the most recent k frames."""
    b, t, c, h, w = h_seq.shape
    if t == k:
        return h_seq
    if t > k:
        return h_seq[:, -k:]
    pad = h_seq[:, :1].expand(b, k - t, c, h, w)
    return torch.cat([pad, h_seq], dim=1)


class _ConvGRUCell(nn.Module):
    """Minimal ConvGRU cell (3x3). Used by adapter_type='convgru_stub'."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv_zr = nn.Conv2d(channels * 2, channels * 2, 3, padding=1)
        self.conv_h = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, x, h):
        zr = torch.sigmoid(self.conv_zr(torch.cat([x, h], dim=1)))
        z, r = zr.chunk(2, dim=1)
        h_hat = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * h_hat


class TemporalHeatmapAdapter(nn.Module):
    def __init__(
        self,
        channels: int,
        window_size: int = 3,
        residual_scale: float = 1.0,
        adapter_type: str = "depthwise_conv3d",
        mix_hidden: int = 128,
    ):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.residual_scale = residual_scale
        self.adapter_type = adapter_type

        if adapter_type == "depthwise_conv3d":
            # per-channel temporal fusion; padding keeps K,h,w
            self.temporal = nn.Conv3d(
                channels, channels, kernel_size=(3, 3, 3),
                padding=(1, 1, 1), groups=channels,
            )
        elif adapter_type == "depthwise_full_window":
            # Fuse the complete fixed-length history and emit one time slice.
            self.temporal = nn.Conv3d(
                channels, channels, kernel_size=(window_size, 3, 3),
                padding=(0, 1, 1), groups=channels,
            )
        elif adapter_type == "convgru_stub":
            self.temporal = _ConvGRUCell(channels)
        else:
            raise ValueError(f"unknown adapter_type {adapter_type!r}")

        # channel mixing: let different kp/line channels exchange structure
        self.mix = nn.Sequential(
            nn.Conv2d(channels, mix_hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mix_hidden, channels, 1),
        )
        # small decoder -> delta
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self._zero_init_last()

    def _zero_init_last(self):
        """Zero the final decoder conv so initial delta == 0 (identity)."""
        last = self.decoder[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def _fuse_time(self, x: torch.Tensor) -> torch.Tensor:
        """[B, K, C, h, w] -> [B, C, h, w] (current-frame temporal context)."""
        if self.adapter_type in {"depthwise_conv3d", "depthwise_full_window"}:
            x = x.permute(0, 2, 1, 3, 4)        # [B, C, K, h, w]
            x = self.temporal(x)                # depthwise over time
            return x[:, :, -1]                  # current/full-window output
        # convgru: roll the window through the cell
        b, k, c, h, w = x.shape
        state = torch.zeros(b, c, h, w, device=x.device, dtype=x.dtype)
        for t in range(k):
            state = self.temporal(x[:, t], state)
        return state

    def forward(self, h_seq: torch.Tensor):
        """h_seq: [B, K, C, h, w] (K == window_size; H_t is the last frame).
        Returns (H_refined, delta), each [B, C, h, w]."""
        h_t = h_seq[:, -1]
        if self.residual_scale == 0:
            return h_t, torch.zeros_like(h_t)
        x = self._fuse_time(h_seq)
        x = self.mix(x)
        delta = self.decoder(x)
        return h_t + self.residual_scale * delta, delta


if __name__ == "__main__":
    # ponytail self-check: shape + zero-init identity + backward.
    B, K, C, h, w = 2, 3, 58, 270, 480
    a = TemporalHeatmapAdapter(channels=C, window_size=K)
    x = torch.randn(B, K, C, h, w)
    ref, d = a(x)
    assert ref.shape == (B, C, h, w) and d.shape == (B, C, h, w)
    assert d.abs().max().item() < 1e-6, "zero-init must give delta==0"
    assert torch.allclose(ref, x[:, -1]), "zero-init must reproduce H_t"
    ref.sum().backward()
    # short window padding
    short = pad_window(torch.randn(B, 1, C, h, w), K)
    assert short.shape == (B, K, C, h, w)
    # convgru path runs
    g = TemporalHeatmapAdapter(channels=24, window_size=K, adapter_type="convgru_stub")
    rg, dg = g(torch.randn(B, K, 24, h, w))
    assert rg.shape == (B, 24, h, w)
    print("ok: shape, zero-init identity, backward, padding, convgru")
