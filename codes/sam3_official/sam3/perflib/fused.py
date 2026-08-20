# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

# NOTE (SoccerNet fine-tuning patch): the upstream public build ships an
# inference-only fused addmm+activation kernel that raises when grad is enabled,
# which makes the ViT-Det backbone (sam3/model/vitdet.py Mlp.forward) untrainable.
# We add a differentiable, non-fused fallback for the grad-enabled (training)
# path. The fused path is unchanged and still used for inference (all inference
# entry points run under torch.inference_mode, i.e. grad disabled), so existing
# detection/tracking benchmarks are byte-for-byte unaffected.
# Original kept at fused.py.orig.

import torch

addmm_act_op = torch.ops.aten._addmm_activation


def _act_apply(activation, y):
    if activation in (torch.nn.functional.relu, torch.nn.ReLU):
        return torch.nn.functional.relu(y)
    if activation in (torch.nn.functional.gelu, torch.nn.GELU):
        return torch.nn.functional.gelu(y)
    raise ValueError(f"Unexpected activation {activation}")


def addmm_act(activation, linear, mat1):
    if torch.is_grad_enabled():
        # Training fallback: differentiable equivalent of the fused kernel.
        # linear is the nn.Linear/nn.Conv2d module (applies weight + bias).
        return _act_apply(activation, linear(mat1))
    self = linear.bias.detach()
    mat2 = linear.weight.detach()
    self = self.to(torch.bfloat16)
    mat1 = mat1.to(torch.bfloat16)
    mat2 = mat2.to(torch.bfloat16)
    mat1_flat = mat1.view(-1, mat1.shape[-1])
    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))
    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))
    raise ValueError(f"Unexpected activation {activation}")
