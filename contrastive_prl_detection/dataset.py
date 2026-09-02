

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

class TrainSet(Dataset):
    def __init__(self, pos_dir, neu_dir, neg_dir, n_patches, withheld_ids=["WITHHELD_SUBJECT_ID",]):
        self.pos_fpaths = sorted([f for f in pos_dir.iterdir() for withheld_id in withheld_ids if withheld_id not in f.name])
        self.neu_fpaths = sorted([f for f in neu_dir.iterdir() for withheld_id in withheld_ids if withheld_id not in f.name])
        self.neg_fpaths = sorted([f for f in neg_dir.iterdir() for withheld_id in withheld_ids if withheld_id not in f.name])

        self.n_patches = n_patches

    def __len__(self):
        return self.n_patches

    def __getitem__(self, _):
        pos = torch.load(np.random.choice(self.pos_fpaths))
        neu = torch.load(np.random.choice(self.neu_fpaths))
        neg = torch.load(np.random.choice(self.neg_fpaths))
        
        return pos, neu, neg