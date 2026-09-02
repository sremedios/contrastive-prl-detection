"""Fully-convolutional 3D residual encoder and whole-volume tiled inference."""

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.c = 2 # lose 2 voxels per axis
        s.f = nn.Sequential(
            nn.Conv3d(i, o, 3, bias=True), nn.ReLU(True),
            nn.Conv3d(o, o, 3, bias=True))
        s.s = nn.Identity() if i == o else nn.Conv3d(i, o, 1, bias=True)
        s.a = nn.ReLU(True)
    def forward(s, x):
        c = s.c
        return s.a(s.f(x) + s.s(x[:, :, c:-c, c:-c, c:-c]))


def resnet3d(c_in=1, c_out=3, w=(16, 32, 64, 128, 256)):
    ch = (c_in,) + tuple(w)
    return nn.Sequential(*[Block(ch[i], ch[i+1]) for i in range(len(w))],
                         nn.Conv3d(w[-1], c_out, 1))


def receptive_field_loss(w):
    """Voxels lost per axis, total, by a `resnet3d` with `len(w)` blocks (4 per block)."""
    return 4 * len(w)


@torch.inference_mode()
def tiled(net, x, tile=64, halo=32):
    B, _, D, H, W = x.shape
    out = None
    for d in range(0, D - halo, tile):
        for h in range(0, H - halo, tile):
            for w in range(0, W - halo, tile):
                y = net(x[:, :, d:d+tile+halo, h:h+tile+halo, w:w+tile+halo])
                if out is None:
                    out = torch.empty(B, y.shape[1], D-halo, H-halo, W-halo,
                                      device=y.device, dtype=y.dtype)
                out[:, :, d:d+y.shape[2], h:h+y.shape[3], w:w+y.shape[4]] = y
    return out
