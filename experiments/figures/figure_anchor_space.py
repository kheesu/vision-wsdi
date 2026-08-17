"""Illustrative view: usages and their visual anchors in one 2-D picture.

This is the intuition figure, not the measurement one (see figure_plane_simple.py
for axes that are literally the numbers the method uses). Text and image
embeddings occupy separate cones of the shared space -- the modality gap -- so
plotting them together raw would park the anchors in a far corner. Subtracting
each modality's own mean removes that offset and lets both sit in one scatter.

The projection is therefore a layout, not a measurement, and the axes carry no
ticks to say so. It is not arbitrary though: in this view the nearest anchor is
the one the method actually picked for 68 of the 69 usages, which the script
recomputes, prints, and refuses to ship below MIN_AGREEMENT.

250 x 200 mm at 300 dpi. The two anchor names, at 24 pt physical poster points,
are the only text in the figure -- the title and the caveat above belong in the
poster body. Dots are coloured by the anchor they were assigned to, which is why
no legend is needed (and why this version shows the mechanism, not correctness).

Run: .venv/bin/python -m experiments.figures.figure_anchor_space
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.decomposition import PCA

from experiments.figures.figure_anchor_plane import ROOT, Word, load, montage

MM = 1 / 25.4
SIZE = (250 * MM, 200 * MM)
DPI = 300
OUT = ROOT / "figures/figure_anchor_space.png"
MIN_AGREEMENT = 0.95   # below this the layout would contradict the method

INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8983"
S1 = "#2a78d6"   # biological cell
S2 = "#eb6834"   # prison cell

# Physical poster points. The anchor names are the figure's only text: the title
# and the explanatory note live in the poster body instead.
FS = {"anchor": 24}

mpl.rcParams.update({
    "font.family": ["Noto Sans", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def photo_at(ax, xy_frac, grid, color, mm=38):
    """The anchor's own photos, `mm` wide on the poster, at an axes-fraction spot.

    OffsetImage's zoom is corrected by dpi/72 internally, so deriving it from a
    physical width keeps the tile the same size whatever DPI the figure uses.
    """
    zoom = mm * MM * 72 / grid.shape[1]
    ax.add_artist(AnnotationBbox(
        OffsetImage(grid, zoom=zoom), xy_frac, xycoords="axes fraction",
        frameon=True, pad=0.16,
        bboxprops=dict(edgecolor=color, linewidth=3.5, facecolor="white"),
        annotation_clip=False, zorder=6))


def main():
    meta, X, proto, targets, classes, occ = load()
    cell = Word("cell", meta, X, proto, targets)
    senses = ["cell.n.02", "cell.n.07"]                  # the two that ever win
    colors = [S1, S2]
    # Per anchor: name, where its photo+name block sits in the side gutter, and
    # which classes/seed give the clearest tiles at poster size.
    keys = [("biological cell", 0.090, 13, 4),
            ("prison cell", 0.910, 4, 2)]
    idx = [cell.senses.index(s) for s in senses]
    # One representative photo prototype per sense: the class that wins most of
    # that sense's assignments.
    anchors = np.stack([proto[cell.top_classes(i)[0]] for i in idx])
    assigned = np.array([idx.index(p) if p in idx else -1 for p in cell.pred])
    assert (assigned >= 0).all(), "a usage was assigned outside the plotted senses"

    usages = cell.t
    # Remove the text/image offset (the modality gap), then project both sets.
    joint = np.vstack([usages - usages.mean(0), anchors - anchors.mean(0)])
    Y = PCA(n_components=2, random_state=0).fit_transform(joint)
    U, V = Y[:len(usages)], Y[len(usages):]

    near = np.linalg.norm(U[:, None, :] - V[None, :, :], axis=2).argmin(1)
    agreement = float((near == assigned).mean())
    assert agreement >= MIN_AGREEMENT, (
        f"the 2-D layout disagrees with the method on {1 - agreement:.0%} of usages")

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_axes([0.020, 0.020, 0.960, 0.960])   # no title or note: full bleed

    # Side gutters: each anchor card (photo + name at 24 pt) needs ~45 mm, so the
    # data is padded into the middle ~60% of the width.
    x0, x1 = min(U[:, 0].min(), V[:, 0].min()), max(U[:, 0].max(), V[:, 0].max())
    pad = 0.32 * (x1 - x0)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(U[:, 1].min() - 0.08 * (x1 - x0), U[:, 1].max() + 0.08 * (x1 - x0))

    for j, color in enumerate(colors):        # spokes: usage -> its anchor
        for p in U[assigned == j]:
            ax.plot([p[0], V[j, 0]], [p[1], V[j, 1]], color=color, lw=1.6,
                    alpha=0.28, zorder=1)
        # Dots take their anchor's colour, so the picture needs no legend.
        sel = assigned == j
        ax.scatter(U[sel, 0], U[sel, 1], s=240, color=color, edgecolor="white",
                   linewidth=2.4, zorder=3)
        ax.scatter(*V[j], marker="*", s=2400, color=color, edgecolor="white",
                   linewidth=2.6, zorder=4)

    for j, ((name, x_frac, seed, k), color) in enumerate(zip(keys, colors)):
        photo_at(ax, (x_frac, 0.66), montage(cell.top_classes(idx[j])[:k],
                                             seed=seed), color)
        ax.text(x_frac, 0.40, name.replace(" ", "\n"), transform=ax.transAxes,
                ha="center", va="top", fontsize=FS["anchor"], color=color,
                fontweight="bold", linespacing=1.15, zorder=6)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=DPI)
    print(f"wrote {OUT} at {SIZE[0] * 25.4:.0f}x{SIZE[1] * 25.4:.0f} mm · "
          f"smallest text {min(FS.values())} pt · "
          f"layout agrees with the argmax on {agreement:.1%} of usages")


if __name__ == "__main__":
    main()
