"""Filesystem layout, volume loading, and the patch Dataset used for training."""

import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

#: Basename pattern for a saved patch: ``{subject_id}-{kind}_patch_{index:04d}.pt``
PATCH_KINDS = ("pos", "neu", "neg")


def get_fpaths(root, subj_id, pos=True):
    fpaths = sorted((Path(root) / subj_id).iterdir())

    pha_fpath = [x for x in fpaths if "_unwrapped.nii" in x.name][0]
    mag_fpath = [x for x in fpaths if "MAG.nii" in x.name][0]
    if pos:
        prl_fpath = [x for x in fpaths if "final_segmentation.nii" in x.name][0]
    else:
        prl_fpath = None
    brainmask_fpath = [x for x in fpaths if "MAG_bet_mask.nii" in x.name][0]
    aultra_fpath = [x for x in fpaths if "reg_separation.nii" in x.name][0]

    return pha_fpath, mag_fpath, prl_fpath, brainmask_fpath, aultra_fpath


def load_ras(fpath):
    # Load, then set up as RAS
    x = nib.load(fpath).get_fdata(dtype=np.float32)
    x = x.transpose(2, 0, 1)
    return torch.from_numpy(x)


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
                 withheld_ids=(), invert=False, seed=None):
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
        self._rng = random.Random(seed)

    def _collect(self, d):
        keep = lambda f: any(w in f.name for w in self.withheld_ids) == self.invert
        return sorted(f for f in Path(d).glob("*.pt") if keep(f))

    def __len__(self):
        return self.n_patches

    def __getitem__(self, _):
        pos = torch.load(self._rng.choice(self.pos_fpaths), weights_only=True)
        neu = torch.load(self._rng.choice(self.neu_fpaths), weights_only=True)
        neg = torch.load(self._rng.choice(self.neg_fpaths), weights_only=True)

        return pos, neu, neg


def worker_init_fn(worker_id):
    """Re-seed each DataLoader worker's `TrainSet` so workers don't duplicate draws."""
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "_rng"):
        info.dataset._rng = random.Random(info.seed)
