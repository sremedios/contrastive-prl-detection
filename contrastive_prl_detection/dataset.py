"""Filesystem layout, volume loading, and the patch Dataset used for training."""

import math
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

#: Basename pattern for a saved patch: ``{subject_id}-{kind}_patch_{index:04d}.pt``
PATCH_KINDS = ("pos", "neu", "neg")

#: Substring that identifies each input volume inside a subject directory.
#: `prl` is only expected for the PRL-positive cohort.
FILE_PATTERNS = {
    "pha": "_unwrapped.nii",
    "mag": "MAG.nii",
    "prl": "final_segmentation.nii",
    "brainmask": "MAG_bet_mask.nii",
    "aultra": "reg_separation.nii",
}
#: Roles returned by `get_fpaths`, in order.
FILE_ROLES = ("pha", "mag", "prl", "brainmask", "aultra")


class IncompleteSubject(Exception):
    """A subject directory is missing one or more of the expected volumes."""

    def __init__(self, root, subj_id, missing, present):
        self.root, self.subj_id = Path(root), subj_id
        self.missing, self.present = dict(missing), list(present)
        wanted = ", ".join(f"{r} (*{p}*)" for r, p in missing.items())
        listing = "\n    ".join(self.present) if self.present else "(directory is empty)"
        super().__init__(
            f"subject {subj_id!r} under {self.root} is missing: {wanted}\n"
            f"  files present:\n    {listing}")


def find_fpaths(root, subj_id, pos=True):
    """Map each role to its file, or to None when nothing matches.

    Substring matching, as in the notebook, so a role with several matches
    resolves to the first in sorted order.
    """
    fpaths = sorted(f for f in (Path(root) / subj_id).iterdir() if f.is_file())
    roles = FILE_ROLES if pos else tuple(r for r in FILE_ROLES if r != "prl")

    found = {}
    for role in FILE_ROLES:
        if role not in roles:
            found[role] = None
            continue
        matches = [x for x in fpaths if FILE_PATTERNS[role] in x.name]
        found[role] = matches[0] if matches else None
    return found


def missing_roles(root, subj_id, pos=True):
    """Roles with no matching file, as {role: pattern}. Empty dict means complete."""
    try:
        found = find_fpaths(root, subj_id, pos=pos)
    except (NotADirectoryError, FileNotFoundError):
        return dict(FILE_PATTERNS)
    roles = FILE_ROLES if pos else tuple(r for r in FILE_ROLES if r != "prl")
    return {r: FILE_PATTERNS[r] for r in roles if found[r] is None}


def get_fpaths(root, subj_id, pos=True):
    """(pha, mag, prl, brainmask, aultra) paths; `prl` is None when `pos=False`.

    Raises `IncompleteSubject` naming the subject and listing what it does
    contain, rather than an opaque IndexError.
    """
    found = find_fpaths(root, subj_id, pos=pos)
    missing = missing_roles(root, subj_id, pos=pos)
    if missing:
        present = sorted(f.name for f in (Path(root) / subj_id).iterdir())
        raise IncompleteSubject(root, subj_id, missing, present)
    return tuple(found[r] for r in FILE_ROLES)


def load_ras(fpath):
    # Load, then set up as RAS
    x = nib.load(fpath).get_fdata(dtype=np.float32)
    x = x.transpose(2, 0, 1)
    return torch.from_numpy(x)


def normalize_intensity(x, lo=-math.pi, hi=math.pi):
    """Linearly rescale a volume's full range onto [lo, hi].

    Magnitude has no absolute scale, so it is put on the same range as phase
    before any patch is cut. Done per volume, never per patch: at inference the
    encoder sweeps a whole volume, and a per-patch rescaling could not be
    reproduced there without breaking the patch/dense equivalence.
    """
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:                       # flat volume; nothing to stretch
        return torch.full_like(x, (lo + hi) / 2)
    return (x - mn) / (mx - mn) * (hi - lo) + lo


def load_mag(fpath, lo=-math.pi, hi=math.pi):
    """Load a magnitude volume and normalise it. Use this everywhere, not load_ras."""
    return normalize_intensity(load_ras(fpath), lo, hi)


def unload_ras(x):
    """Inverse of `load_ras`'s axis permutation, for writing results back out."""
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return x.transpose(1, 2, 0)


def list_subject_ids(root):
    """Sorted ids of the subject directories directly under `root`."""
    return sorted(x.name for x in Path(root).iterdir()
                  if x.is_dir() and not x.name.startswith("."))


def patch_fname(subj_id, kind, index):
    return f"{subj_id}-{kind}_patch_{index:04d}.pt"


def subject_id_of(fpath):
    """Recover the subject id from a patch filename written by `patch_fname`."""
    return Path(fpath).name.rsplit("-", 1)[0]


def subject_ids_in(patch_dir):
    """Sorted subject ids that have at least one patch in `patch_dir`."""
    return sorted({subject_id_of(f) for f in Path(patch_dir).glob("*.pt")})


def random_octahedral(x, rng):
    """A random element of the cube's symmetry group: 90 deg turns and flips.

    Index permutations and reversals only, so no interpolation and no resampling
    blur; `x` is (C, D, H, W) and the channel axis is left alone.
    """
    for dims in ((1, 2), (1, 3), (2, 3)):
        k = rng.randrange(4)
        if k:
            x = torch.rot90(x, k, dims=dims)
    for d in (1, 2, 3):
        if rng.random() < 0.5:
            x = torch.flip(x, dims=(d,))
    return x.contiguous()


class TrainSet(Dataset):
    """Draws one (positive, neutral, negative) patch triplet per item.

    `__len__` is a virtual epoch length: every `__getitem__` picks a fresh random
    file from each class pool, so `n_patches` sets how many triplets one pass
    over the loader yields rather than how many distinct patches exist.

    `withheld_ids` implements the leave-subjects-out split.  With `invert=False`
    (the default) those subjects are *excluded*, which is the training half; with
    `invert=True` only those subjects are kept, which is the validation half.
    """

    def __init__(self, pos_dir, neu_dir, neg_dir, n_patches,
                 withheld_ids=(), invert=False, seed=None, augment=False):
        self.withheld_ids = tuple(withheld_ids)
        self.invert = invert

        self.pos_fpaths = self._collect(pos_dir)
        self.neu_fpaths = self._collect(neu_dir)
        self.neg_fpaths = self._collect(neg_dir)

        for kind, fpaths, d in (("positive", self.pos_fpaths, pos_dir),
                                ("neutral", self.neu_fpaths, neu_dir),
                                ("negative", self.neg_fpaths, neg_dir)):
            if not fpaths:
                raise ValueError(
                    f"no {kind} patches left in {d} after applying "
                    f"withheld_ids={self.withheld_ids!r} (invert={invert})")

        self.n_patches = n_patches
        self.augment = augment
        self._rng = random.Random(seed)

    def _collect(self, d):
        keep = lambda f: any(w in f.name for w in self.withheld_ids) == self.invert
        return sorted(f for f in Path(d).glob("*.pt") if keep(f))

    def __len__(self):
        return self.n_patches

    def _draw(self, fpaths):
        x = torch.load(self._rng.choice(fpaths), weights_only=True)
        return random_octahedral(x, self._rng) if self.augment else x

    def __getitem__(self, _):
        return (self._draw(self.pos_fpaths),
                self._draw(self.neu_fpaths),
                self._draw(self.neg_fpaths))


def worker_init_fn(worker_id):
    """Re-seed each DataLoader worker's `TrainSet` so workers don't duplicate draws."""
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "_rng"):
        info.dataset._rng = random.Random(info.seed)
