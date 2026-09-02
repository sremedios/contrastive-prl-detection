def get_patch_center(tup):
    x = (tup[0].stop + tup[0].start)//2
    y = tup[1].start # Annotation is only 2D so we just take the start here
    z = (tup[2].stop + tup[2].start)//2
    return x, y, z

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

def get_neu_patches(mag, pha, seg, brainmask, patch_size, n, frac_brain=0.95, rng=None):
    rng = rng or np.random.default_rng()
    # Centers whose patch contains no seg voxel at all
    empty = torch.from_numpy(ndimage.maximum_filter(np.asarray(seg > 0.5), size=patch_size) == 0)
    brain = brainmask > 0.5

    n_bg = int(round(n * (1 - frac_brain)))
    n_brain = n - n_bg

    centers = sample_centers(empty & brain, patch_size, n_brain, rng)
    centers += sample_centers(empty & ~brain, patch_size, n_bg, rng)
    return get_patches(mag, pha, centers, patch_size)

def get_pos_patches(mag, pha, prl, patch_size):
    labels, _ = ndimage.label(prl > 0.5, structure=ndimage.generate_binary_structure(3, 3))
    centers = [get_patch_center(s) for s in ndimage.find_objects(labels)]
    return get_patches(mag, pha, centers, patch_size)