"""Extraction of (magnitude, phase) training patches from a volume.

Every patch is a `(2, *patch_size)` tensor stacked as `[magnitude, phase]`, in
the RAS-ish orientation produced by `dataset.load_ras`.
"""

import numpy as np
import torch
from scipy import ndimage


def get_patch_center(tup):
    """Midpoint of a component's bounding box, on every axis.

    The old rule took `.start` on axis 1, on the assumption that the 2D
    annotation made that axis degenerate. `load_ras` transposes (2, 0, 1), so
    axis 1 is not the slice axis and the patch was offset by half a lesion width
    along it -- an offset the negative patches do not have.

    The midpoint is correct whichever axis is the degenerate one: for an extent
    of 1, `(start + stop) // 2 == start`, so this needs no assumption about the
    annotation plane. `bbox_extents` reports the extents to confirm it.
    """
    return tuple((sl.start + sl.stop) // 2 for sl in tup)


def bbox_extents(tup):
    """Per-axis extent of a component's bounding box. Extent 1 = the 2D axis."""
    return tuple(sl.stop - sl.start for sl in tup)

def get_patch_coords(cs, ps):
    return tuple(slice(c - p//2, c - p//2 + p) for c, p in zip(cs, ps))

def get_patches(mag, pha, centers, patch_size):
    patch_coords = [get_patch_coords(c, ps=patch_size) for c in centers]
    xs_pha = torch.stack([pha[coord] for coord in patch_coords])
    xs_mag = torch.stack([mag[coord] for coord in patch_coords])
    return torch.stack([xs_mag, xs_pha], dim=1)

def sample_centers(mask, patch_size, n, rng=None):
    rng = rng or np.random.default_rng()
    inbounds = torch.zeros(mask.shape, dtype=bool)
    inbounds[tuple(slice(p // 2, s - p // 2) for s, p in zip(mask.shape, patch_size))] = True
    idx = torch.argwhere(mask & inbounds)
    sel = rng.choice(len(idx), size=min(n, len(idx)), replace=False)
    return [tuple(c) for c in idx[sel]]

def get_neg_patches(mag, pha, seg, patch_size, n, rng=None):
    centers = sample_centers(seg > 0.5, patch_size, n, rng)
    return get_patches(mag, pha, centers, patch_size)

def get_neu_patches(mag, pha, seg, brainmask, patch_size, n, rng=None):
    """Lesion-free patches, every centre inside the brain mask.

    Centres used to come 95% from inside the mask and the rest from outside it,
    to show the encoder some air. The mask is present for every subject, so
    whatever the encoder does outside it is discarded downstream and never
    read; those patches only spent capacity on background, and are dropped.
    """
    if seg.shape != brainmask.shape:
        raise ValueError(f"segmentation {tuple(seg.shape)} and brain mask "
                         f"{tuple(brainmask.shape)} are on different grids")
    # Centers whose patch contains no seg voxel at all
    empty = torch.from_numpy(ndimage.maximum_filter(np.asarray(seg > 0.5), size=patch_size) == 0)
    centers = sample_centers(empty & (brainmask > 0.5), patch_size, n, rng)
    return get_patches(mag, pha, centers, patch_size)

def get_pos_patches(mag, pha, prl, patch_size, return_extents=False):
    labels, _ = ndimage.label(prl > 0.5, structure=ndimage.generate_binary_structure(3, 3))
    boxes = ndimage.find_objects(labels)
    centers = [get_patch_center(b) for b in boxes]
    keep = [i for i, c in enumerate(centers) if _inbounds(c, prl.shape, patch_size)]
    patches = get_patches(mag, pha, [centers[i] for i in keep], patch_size)
    if return_extents:
        return patches, [bbox_extents(boxes[i]) for i in keep]
    return patches


def _inbounds(center, shape, patch_size):
    """True if the patch around `center` fits entirely inside `shape`."""
    return all(c - p // 2 >= 0 and c - p // 2 + p <= s
               for c, p, s in zip(center, patch_size, shape))
