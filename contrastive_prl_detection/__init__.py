"""Contrastive PRL detection.

A 3D residual encoder is trained to map (magnitude, phase) patches onto the unit
circle S^1, with three classes pinned to anchor directions 120 degrees apart:

    positive (+)  ->   90 deg   (paramagnetic rim lesion)
    neutral  (~)  ->  210 deg   (no lesion)
    negative (-)  ->  330 deg   (rim-negative lesion)

Because the encoder is fully convolutional, the same weights that classify a
patch can be swept over a whole volume to produce a continuous theta-map.
"""

from . import dataset, net, patch_utils, polar_utils, contrastive  # noqa: F401

__version__ = "0.1.0"
