import numpy as np
from matplotlib.colors import ListedColormap, to_rgb

_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_W = np.array([0.95047, 1.0, 1.08883])


cmap = cyclic3("#86101E", "#000000", "#4C8FD4")   # build once, outside the loop


def _rgb2lab(rgb):
    x = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4) @ _M.T / _W
    f = np.where(x > (6 / 29) ** 3, np.cbrt(x), x * 841 / 108 + 4 / 29)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def _lab2rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    lin = (np.where(f > 6 / 29, f ** 3, (f - 4 / 29) * 108 / 841) * _W) @ np.linalg.inv(_M).T
    srgb = np.where(lin <= 0.0031308, 12.92 * lin,
                    1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055)
    return np.clip(srgb, 0, 1)


def cyclic3(c1, c2, c3, angles=(90, 210, 330), N=256, smooth=5, name="cyclic3"):
    """Build a cyclic colormap with c1, c2, c3 at the given data angles."""
    lab = _rgb2lab(np.array([to_rgb(c) for c in (c1, c2, c3)]))
    L, C = lab[:, 0], np.hypot(lab[:, 1], lab[:, 2])
    h = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
    for i in np.where(C < 3)[0]:                       # neutral: no meaningful hue,
        j = (i + 1) % 3 if C[(i + 1) % 3] >= 3 else (i - 1) % 3
        h[i] = h[j]                                    # borrow a neighbour's
    a = np.asarray(angles, float) % 360
    if not 0 < (a[1] - a[0]) % 360 < (a[2] - a[0]) % 360:
        raise ValueError("angles must be in cyclic order")

    u = np.array([0.0, (a[1] - a[0]) % 360, (a[2] - a[0]) % 360, 360.0])
    hop = (np.diff(h, append=h[0]) + 180) % 360 - 180  # short way round, per hop
    node = lambda v, wrap: np.append(v, wrap)
    t = (np.arange(N) / N * 360 - a[0]) % 360
    Lv = np.interp(t, u, node(L, L[0]))
    Cv = np.interp(t, u, node(C, C[0]))
    hv = np.interp(t, u, np.append(h[0] + np.append(0, np.cumsum(hop)[:2]),
                                   h[0] + hop.sum()))

    if smooth:                                         # cyclic Gaussian on L only
        k = np.exp(-0.5 * (np.arange(-4 * smooth, 4 * smooth + 1) / smooth) ** 2)
        Lv = np.convolve(np.tile(Lv, 3), k / k.sum(), "same")[N:2 * N]

    rgb = _lab2rgb(np.stack([Lv, Cv * np.cos(np.radians(hv)),
                             Cv * np.sin(np.radians(hv))], -1))
    return ListedColormap(rgb, name=name)


def rotate_theta(theta, ref_deg=210.0):
    """Shift so the class at ref_deg lands on twilight's white seam (+/- pi)."""
    return np.angle(np.exp(1j * (theta - np.radians(ref_deg) + np.pi)))

def circular_colorbar(ax, anchors_deg=(90.0, 210.0, 330.0),
                      names=("pos", "neg", "neu"), cmap="twilight",
                      r_in=0.62, r_out=1.0):
    t = np.linspace(0, 2*np.pi, 721)
    r = np.linspace(r_in, r_out, 2)
    T, R = np.meshgrid(t, r)
    ax.pcolormesh(T, R, T, cmap=cmap, vmin=0, vmax=2*np.pi,
                  shading="nearest", rasterized=True)
    for a_deg, nm in zip(anchors_deg, names):
        a = np.radians(a_deg)
        ax.plot([a, a], [r_in, r_out], color="k", lw=1.4, zorder=3)
        ax.text(a, r_out * 1.30, nm, ha="center", va="center", fontsize=10)
    for b_deg in (150.0, 270.0, 30.0):
        b = np.radians(b_deg)
        ax.plot([b, b], [r_in, r_out], color="w", lw=1.0, ls=":", zorder=3)
    ax.set_ylim(0, r_out * 1.15)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    return ax