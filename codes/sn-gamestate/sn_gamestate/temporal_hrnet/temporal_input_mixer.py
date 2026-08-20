"""Temporal input mixer for frozen NBJW HRNet.

This is the smallest upstream temporal hook: K RGB frames in, one RGB frame out.
The 1x1 conv is initialized to exactly copy the center/current frame, so a fresh
checkpoint is baseline-identical before training.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalInputMixer(nn.Module):
    """Fuse RGB frame windows before HRNet.

    Input:  [B, K, 3, H, W]
    Output: [B, 3, H, W]
    """

    def __init__(self, window_size: int = 3):
        super().__init__()
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self.proj = nn.Conv2d(window_size * 3, 3, kernel_size=1)
        self.reset_identity()

    def reset_identity(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        center = self.window_size - 1
        with torch.no_grad():
            for channel in range(3):
                self.proj.weight[channel, center * 3 + channel, 0, 0] = 1.0

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(f"expected [B,K,3,H,W], got {tuple(frames.shape)}")
        b, k, c, h, w = frames.shape
        if k != self.window_size or c != 3:
            raise ValueError(f"expected K={self.window_size}, C=3, got K={k}, C={c}")
        return self.proj(frames.reshape(b, k * c, h, w))


if __name__ == "__main__":
    x = torch.randn(2, 3, 3, 16, 16)
    mixer = TemporalInputMixer(window_size=3)
    y = mixer(x)
    assert y.shape == (2, 3, 16, 16)
    assert torch.allclose(y, x[:, -1])
    y.sum().backward()
    print("ok: temporal input mixer identity + backward")
