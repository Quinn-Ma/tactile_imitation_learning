"""Small adapted subset of TD-MPC2's MIT-licensed common/layers.py.

Upstream: https://github.com/nicklashansen/tdmpc2
Snapshot: e9f5932. Copyright (c) Nicklas Hansen (2023).
Changes: no TensorDict dependencies; augmentation disabled in evaluation;
configuration replaced by explicit arguments; no mutable-default encoder factory.
See LICENSE_TDMPC2.txt. This is not the complete TD-MPC2 algorithm.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SimNorm(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        shape = x.shape
        return F.softmax(x.reshape(*shape[:-1], -1, self.dim), dim=-1).reshape(shape)


class NormedLinear(nn.Linear):
    def __init__(self, *args, act=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = nn.Mish(inplace=False) if act is None else act

    def forward(self, x):
        return self.act(self.ln(super().forward(x)))


def mlp(in_dim, hidden, out_dim, act=None):
    dimensions = [in_dim] + list(hidden) + [out_dim]
    modules = [NormedLinear(a, b) for a, b in zip(dimensions[:-2], dimensions[1:-1])]
    modules.append(NormedLinear(dimensions[-2], dimensions[-1], act=act)
                   if act is not None else nn.Linear(dimensions[-2], dimensions[-1]))
    return nn.Sequential(*modules)


class ShiftAug(nn.Module):
    def __init__(self, pad=3):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        if not self.training or self.pad == 0:
            return x
        n, _, h, w = x.shape
        if h != w:
            raise ValueError('TD-MPC2 shift augmentation expects square images.')
        padded = F.pad(x, (self.pad,) * 4, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        axis = torch.linspace(-1 + eps, 1 - eps, h + 2 * self.pad,
                              device=x.device, dtype=x.dtype)[:h]
        xx = axis.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        grid = torch.cat((xx, xx.transpose(0, 1)), 2).unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, (n, 1, 1, 2), device=x.device)
        grid = grid + shift.to(x.dtype) * (2.0 / (h + 2 * self.pad))
        return F.grid_sample(padded, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
    def forward(self, x):
        return x.float().div(255.0).sub(0.5)


def conv(in_channels=9, num_channels=16):
    """Official four-convolution pattern; 64x64 -> 16*num_channels features."""
    return nn.Sequential(
        ShiftAug(), PixelPreprocess(),
        nn.Conv2d(in_channels, num_channels, 7, stride=2), nn.ReLU(),
        nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(),
        nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(),
        nn.Conv2d(num_channels, num_channels, 3), nn.Flatten(), SimNorm(8))
