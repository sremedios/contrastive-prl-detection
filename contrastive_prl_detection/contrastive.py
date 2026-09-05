"""The S^1 objective: anchors, logits, and angle decoding.

The encoder emits an unnormalised vector u in R^2.  Projecting it onto the unit
circle (z = u/||u||) discards the radius and keeps only the angle theta, and the
class scores are the cosine similarities between z and three fixed anchor
directions, divided by a temperature.  Training is then plain cross-entropy.
"""

import math

import numpy as np
import torch
from torch.nn import functional as F

#: Anchor directions in degrees, in class order (positive, neutral, negative).
ANCHORS_DEG = (90.0, 210.0, 330.0)
#: Bisectors between the anchors, i.e. the decision boundaries.
BISECTORS_DEG = (150.0, 270.0, 30.0)
#: Default softmax temperature.
TAU = 0.2

CLASS_NAMES = ("positive (+)", "neutral (~)", "negative (-)")
CLASS_COLORS = ("#d1495b", "#5b6c8f", "#2a9d8f")


def make_anchors(device=None, dtype=torch.float32, anchors_deg=ANCHORS_DEG):
    """Three unit vectors 120 deg apart on S^1. Order: pos, neu, neg."""
    ang = torch.tensor(anchors_deg, device=device, dtype=dtype) * math.pi / 180
    return torch.stack([ang.cos(), ang.sin()], dim=1)          # (3, 2)


def project(u, dim=1):
    """Project the 2-vectors living on axis `dim` onto S^1."""
    return F.normalize(u, dim=dim, eps=1e-6)


def logits(z, anchors, tau=TAU, dim=1):
    """Cosine similarity of `z` to each anchor, over temperature.

    Works for both patch output `(N, 2)` with `dim=1` and dense volume output
    `(B, 2, D, H, W)` with `dim=1`; the 2-channel axis becomes a 3-class axis.
    """
    sim = torch.movedim(z, dim, -1) @ anchors.T
    return torch.movedim(sim, -1, dim) / tau


#: Weight on the unit-norm penalty added to the cross-entropy. See `norm_penalty`.
LAMBDA_NORM = 0.1


def norm_penalty(u, dim=1):
    """Mean squared deviation of ||u|| from 1, pulling the raw output onto S^1.

    Cross-entropy alone says nothing about the radius: the Jacobian of `project`
    is orthogonal to u, so no gradient ever reaches ||u||, which is then left to
    drift on initialisation and weight decay alone. This penalty is the only
    term that sets it.

    Bounding the radius changes what the disc is used for. Unpenalised, the
    encoder parks patches far out where the radius carries nothing; held near 1,
    it starts using the interior, and ||u|| becomes a margin worth reading --
    patches the model is unsure of sit in toward the origin instead of at an
    arbitrary large radius. `plot_both_views` panel (a) is where that shows up.
    """
    return (u.norm(dim=dim) - 1).pow(2).mean()


def theta(z, dim=1):
    """Angle of each 2-vector on axis `dim`, in (-pi, pi]."""
    zx, zy = z.unbind(dim)
    return torch.atan2(zy, zx)


def theta_np(u):
    """Same as `project` + `theta` for a numpy `(N, 2)` array."""
    z = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)
    return z, np.arctan2(z[:, 1], z[:, 0])


@torch.no_grad()
def embed(model, loader, device):
    """Returns both the raw R^2 output and its projection onto S^1."""
    model.eval()
    U, Y = [], []
    for pos, neu, neg in loader:
        x = torch.cat([pos, neu, neg], dim=0).to(device)
        n = pos.shape[0]
        Y.append(torch.arange(3).repeat_interleave(n))
        U.append(model(x).flatten(1).cpu())
    u, y = torch.cat(U).numpy(), torch.cat(Y).numpy()
    z, th = theta_np(u)
    return u, z, y, th


def accuracy_from_theta(th, y, anchors_deg=ANCHORS_DEG):
    """Per-class and overall accuracy of the nearest-anchor decision rule."""
    a = np.radians(anchors_deg)
    d = np.abs(np.angle(np.exp(1j * (th[:, None] - a[None, :]))))   # (N, 3)
    pred = d.argmin(1)
    per_class = [float((pred[y == c] == c).mean()) if (y == c).any() else float("nan")
                 for c in range(len(anchors_deg))]
    return float((pred == y).mean()), per_class
