"""Cyclic colormaps and plots for the S^1 embedding.

`theta` is an angle, so a linear colormap would put a false seam somewhere on the
circle.  `cyclic3` builds a perceptually-even cyclic map that pins one colour to
each class anchor, so a theta-map can be read directly as a class map.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb

from .contrastive import (ANCHORS_DEG, BISECTORS_DEG, CLASS_COLORS,
                          CLASS_NAMES)

_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_W = np.array([0.95047, 1.0, 1.08883])


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


#: Default theta colormap: red at the positive anchor, black at neutral, blue at
#: negative.  Built once at import so plotting loops don't rebuild it.
cmap = cyclic3("#86101E", "#000000", "#4C8FD4")



def rotate_theta(theta, ref_deg=210.0):
    """Shift so the class at ref_deg lands on twilight's white seam (+/- pi)."""
    return np.angle(np.exp(1j * (theta - np.radians(ref_deg) + np.pi)))

def circular_colorbar(ax, anchors_deg=ANCHORS_DEG,
                      names=("pos", "neu", "neg"), cmap=None,
                      r_in=0.62, r_out=1.0):
    """Draw the theta legend as an annulus. `ax` must be a polar axes."""
    cmap = globals()["cmap"] if cmap is None else cmap
    t = np.linspace(0, 2*np.pi, 721)
    r = np.linspace(r_in, r_out, 2)
    T, R = np.meshgrid(t, r)
    ax.pcolormesh(T, R, T, cmap=cmap, vmin=0, vmax=2*np.pi,
                  shading="nearest", rasterized=True)
    # shading="nearest" centres cells on the given radii, so the mesh actually
    # reaches half a cell beyond r_out. Labels are placed past that true edge,
    # not past r_out, or they sit on top of the ring.
    r_edge = r_out + (r_out - r_in) / 2
    for a_deg, nm in zip(anchors_deg, names):
        a = np.radians(a_deg)
        ax.plot([a, a], [r_in, r_out], color="k", lw=1.4, zorder=3)
        ax.text(a, r_edge * 1.18, nm, ha="center", va="center", fontsize=10)
    for b_deg in BISECTORS_DEG:
        b = np.radians(b_deg)
        ax.plot([b, b], [r_in, r_out], color="w", lw=1.0, ls=":", zorder=3)
    ax.set_ylim(0, r_edge * 1.34)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    return ax

def plot_both_views(u, z, y, theta, anchors_deg=ANCHORS_DEG,
                    rng=None, show=True, savepath=None, close=True):
    """Side-by-side: raw encoder output in R^2, and the same points on S^1."""
    rng = np.random.default_rng(0) if rng is None else rng
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    anchors_rad = np.radians(anchors_deg)
    bisectors   = np.radians(BISECTORS_DEG)

    # ---- (a) raw encoder output in R^2 ----
    ax = axes[0]
    lim = np.percentile(np.linalg.norm(u, axis=1), 99) * 1.15
    for b in bisectors:                                  # decision boundaries
        ax.plot([0, lim * 1.5 * np.cos(b)], [0, lim * 1.5 * np.sin(b)],
                ls=":", lw=1.0, c="#999999", zorder=0)
    for c, a in enumerate(anchors_rad):                  # anchor directions
        ax.plot([0, lim * 1.5 * np.cos(a)], [0, lim * 1.5 * np.sin(a)],
                ls="--", lw=1.0, c=CLASS_COLORS[c], alpha=0.55, zorder=0)
    for c in range(3):
        m = y == c
        ax.scatter(u[m, 0], u[m, 1], s=9, c=CLASS_COLORS[c],
                   alpha=0.7, linewidths=0, label=CLASS_NAMES[c])
    ax.scatter([0], [0], s=28, c="k", marker="+", zorder=6)
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("$u_1$"); ax.set_ylabel("$u_2$")
    ax.set_title("(a) raw encoder output $u \\in \\mathbb{R}^2$\n"
                 "dashed = anchor directions, dotted = decision boundaries",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # ---- (b) same points, radially projected onto S^1 ----
    ax = axes[1]
    circ = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(circ), np.sin(circ), color="#cccccc", lw=1, zorder=0)
    for b in bisectors:
        ax.plot([0, 1.35 * np.cos(b)], [0, 1.35 * np.sin(b)],
                ls=":", lw=1.0, c="#bbbbbb", zorder=0)
    jitter = 1 + 0.045 * rng.standard_normal(len(theta))   # viz only
    for c in range(3):
        m = y == c
        ax.scatter(jitter[m] * np.cos(theta[m]), jitter[m] * np.sin(theta[m]),
                   s=9, c=CLASS_COLORS[c], alpha=0.7, linewidths=0)
    for c, a in enumerate(anchors_rad):
        ax.scatter(np.cos(a), np.sin(a), s=190, marker="*", c=CLASS_COLORS[c],
                   edgecolors="k", linewidths=0.9, zorder=5)
    ax.set_aspect("equal"); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(b) after $z = u/\\|u\\|$: same angles, radius discarded\n"
                 "(radial jitter for visibility only)", fontsize=10)

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    elif close:
        plt.close(fig)
    return fig


def plot_theta_slices(y_hat, slices, overlay=None, ncols=4, figscale=5, dpi=200,
                      title=None, overlay_color="#ffd400", overlay_lw=1.0,
                      show=True, savepath=None, close=True):
    """Grid of axial theta-map slices, optionally outlining a lesion mask.

    `y_hat` is the theta-map wrapped into [0, 2pi) and `slices` indexes its last
    axis, matching the orientation `dataset.load_ras` produces. The overlay is
    drawn as a contour rather than a translucent fill, so the theta values it
    marks stay readable underneath it.

    Constrained layout, not tight_layout: it accounts for `suptitle`, which
    tight_layout overlaps with the top row.
    """
    slices = list(slices)
    ncols = min(ncols, len(slices))
    nrows = int(np.ceil(len(slices) / ncols))
    h, w = y_hat.shape[0], y_hat.shape[1]
    figsize = (figscale * ncols * h / w, figscale * nrows)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                            dpi=dpi, constrained_layout=True)

    for ax, sl_idx in zip(axs.flat, slices):
        ax.imshow(y_hat[..., sl_idx].T, cmap=cmap, vmin=0, vmax=2 * np.pi,
                  interpolation="nearest")
        if overlay is not None:
            m = np.asarray(overlay[..., sl_idx]).T
            if (m > 0.5).any():   # contour warns on an all-empty slice
                ax.contour(m, levels=[0.5], colors=[overlay_color],
                           linewidths=overlay_lw)
        ax.set_title(f"slice {sl_idx}", fontsize=8)
    for ax in axs.flat:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    elif close:
        plt.close(fig)
    return fig


def plot_circular_colorbar(anchors_deg=ANCHORS_DEG, names=("pos", "neu", "neg"),
                           figscale=3.2, dpi=200, title=None,
                           show=True, savepath=None, close=True):
    """Standalone theta legend: the cyclic colormap drawn as an annulus.

    A key for `plot_theta_slices`, whose colours encode an angle and so cannot
    be read off a linear colorbar.
    """
    fig, ax = plt.subplots(figsize=(figscale, figscale), dpi=dpi,
                           subplot_kw={"projection": "polar"},
                           constrained_layout=True)
    circular_colorbar(ax, anchors_deg=anchors_deg, names=names)
    if title:
        ax.set_title(title, fontsize=10, pad=14)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    elif close:
        plt.close(fig)
    return fig
