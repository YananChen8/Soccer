"""Tiny trainable multi-frame fusion inside frozen NBJW HRNet.

ponytail: this wrapper copies the existing HRNet forward instead of refactoring
the vendor model. Delete it if cls_hrnet.py ever exposes encode/head methods.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFeatureFusion(nn.Module):
    """Identity-initialized residual fusion for [B, K, C, H, W] features."""

    def __init__(self, channels: int, window_size: int = 3, hidden: int = 64, scale: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.scale = scale
        self.temporal = nn.Conv3d(
            channels, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), groups=channels
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, min(hidden, channels), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(min(hidden, channels), channels, 1),
        )
        nn.init.zeros_(self.proj[-1].weight)
        if self.proj[-1].bias is not None:
            nn.init.zeros_(self.proj[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        current = features[:, -1]
        x = features.permute(0, 2, 1, 3, 4)
        delta = self.proj(self.temporal(x)[:, :, -1])
        return current + self.scale * delta


class LoRAConv2d(nn.Module):
    """Frozen Conv2d plus tiny trainable low-rank residual."""

    def __init__(self, base: nn.Conv2d, rank: int = 4, scale: float = 0.05):
        super().__init__()
        self.base = base
        self.scale = scale
        self.down = nn.Conv2d(base.in_channels, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.up(self.down(x))


class TemporalHRNetFeatureFusion(nn.Module):
    """Fuse conv1, stage1, or pre-head stage4 features of one HRNet."""

    def __init__(
        self,
        hrnet: nn.Module,
        level: str = "last",
        window_size: int = 3,
        residual_scale: float = 1.0,
        head_lora_rank: int = 0,
        head_lora_scale: float = 0.05,
        freeze_hrnet: bool = True,
    ):
        super().__init__()
        if level not in {"first", "stage1", "last"}:
            raise ValueError(f"unknown fusion level: {level}")
        self.hrnet = hrnet
        self.level = level
        self.window_size = window_size
        channels = {"first": 64, "stage1": 256, "last": 720}[level]
        hidden = 64 if level in {"first", "stage1"} else 128
        self.fusion = TemporalFeatureFusion(channels, window_size=window_size, hidden=hidden, scale=residual_scale)
        if freeze_hrnet:
            for parameter in self.hrnet.parameters():
                parameter.requires_grad_(False)
        self.head_lora_rank = head_lora_rank
        if head_lora_rank > 0:
            self.hrnet.head[0][0] = LoRAConv2d(self.hrnet.head[0][0], rank=head_lora_rank, scale=head_lora_scale)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def adapter_state_dict(self):
        return {
            key: value
            for key, value in self.state_dict().items()
            if key.startswith("fusion.") or ".down." in key or ".up." in key
        }

    def load_adapter_state_dict(self, state_dict):
        if any(key.startswith("fusion.") for key in state_dict):
            self.load_state_dict(state_dict, strict=False)
        else:
            self.fusion.load_state_dict(state_dict)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, K, 3, H, W], current frame is frames[:, -1].
        if self.level == "first":
            conv1 = [self.hrnet.conv1(frames[:, i]) for i in range(frames.size(1))]
            x_skip = self.fusion(torch.stack(conv1, dim=1))
            x = self._encode_after_conv1(x_skip)
            return self.hrnet._make_head(x, x_skip)

        if self.level == "stage1":
            skips, stage1 = [], []
            for i in range(frames.size(1)):
                x_skip = self.hrnet.conv1(frames[:, i])
                skips.append(x_skip)
                stage1.append(self._stem_to_stage1(x_skip))
            x = self.fusion(torch.stack(stage1, dim=1))
            return self.hrnet._make_head(self._stages_to_head_feature(x), skips[-1])

        skips, feats = [], []
        for i in range(frames.size(1)):
            x_skip, feat = self._encode_to_head(frames[:, i])
            skips.append(x_skip)
            feats.append(feat)
        feat = self.fusion(torch.stack(feats, dim=1))
        return self.hrnet._make_head(feat, skips[-1])

    def _encode_after_conv1(self, x_skip: torch.Tensor) -> torch.Tensor:
        return self._stages_to_head_feature(self._stem_to_stage1(x_skip))

    def _stem_to_stage1(self, x_skip: torch.Tensor) -> torch.Tensor:
        x = self.hrnet.bn1(x_skip)
        x = self.hrnet.relu(x)
        x = self.hrnet.conv2(x)
        x = self.hrnet.bn2(x)
        x = self.hrnet.relu(x)
        return self.hrnet.layer1(x)

    def _encode_to_head(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_skip = self.hrnet.conv1(image)
        return x_skip, self._encode_after_conv1(x_skip)

    def _stages_to_head_feature(self, x: torch.Tensor) -> torch.Tensor:
        x_list = []
        for i in range(self.hrnet.stage2_cfg["NUM_BRANCHES"]):
            x_list.append(self.hrnet.transition1[i](x) if self.hrnet.transition1[i] is not None else x)
        y_list = self.hrnet.stage2(x_list)

        x_list = []
        for i in range(self.hrnet.stage3_cfg["NUM_BRANCHES"]):
            x_list.append(self.hrnet.transition2[i](y_list[-1]) if self.hrnet.transition2[i] is not None else y_list[i])
        y_list = self.hrnet.stage3(x_list)

        x_list = []
        for i in range(self.hrnet.stage4_cfg["NUM_BRANCHES"]):
            x_list.append(self.hrnet.transition3[i](y_list[-1]) if self.hrnet.transition3[i] is not None else y_list[i])
        x = self.hrnet.stage4(x_list)

        height, width = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(height, width), mode="bilinear", align_corners=False)
        x2 = F.interpolate(x[2], size=(height, width), mode="bilinear", align_corners=False)
        x3 = F.interpolate(x[3], size=(height, width), mode="bilinear", align_corners=False)
        return torch.cat([x[0], x1, x2, x3], 1)


if __name__ == "__main__":
    class FakeHRNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(64)

    x = torch.randn(2, 3, 64, 8, 8)
    f = TemporalFeatureFusion(64, 3)
    assert torch.allclose(f(x), x[:, -1], atol=1e-6)
    print("ok: temporal feature fusion identity")
