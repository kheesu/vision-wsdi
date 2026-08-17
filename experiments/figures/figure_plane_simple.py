"""The anchor plane as a single plain matplotlib scatter, for the poster.

Same data and same axes as figure_anchor_plane.py -- the two coordinates are the
cosines the method actually computes, so the diagonal y = x IS the decision rule
and nothing is projected -- but with the photo montages, callouts and prose
stripped out. 200 x 170 mm at 300 dpi; type sizes are physical poster points.

Run: .venv/bin/python -m experiments.figures.figure_plane_simple
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from experiments.figures.figure_anchor_plane import ROOT, RUN, Word, load

MM = 1 / 25.4
SIZE = (200 * MM, 170 * MM)
DPI = 300
OUT = ROOT / "figures/figure_plane_simple.png"

INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8983"
GRID = "#e6e5e1"
S1 = "#2a78d6"   # biological cell
S2 = "#eb6834"   # prison cell

mpl.rcParams.update({
    "font.family": ["Noto Sans", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK2,
    "xtick.color": INK3,
    "ytick.color": INK3,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "savefig.facecolor": "white",
})


def main():
    meta, X, proto, targets, classes, occ = load()
    cell = Word("cell", meta, X, proto, targets)
    bio = cell.senses.index("cell.n.02")
    prison = cell.senses.index("cell.n.07")
    # The 2-D boundary must reproduce the real argmax over all candidate senses.
    two = np.where(cell.score[:, bio] > cell.score[:, prison], bio, prison)
    assert (two == cell.pred).all(), "the diagonal would misstate the decision"

    metrics = pd.read_csv(RUN / "metrics.csv")
    null = metrics[(metrics.lemma == "cell")
                   & (metrics.method == "anchor-assignment-shuffled")].ari

    x, y = cell.score[:, bio], cell.score[:, prison]
    lo, hi = 0.16, 0.62

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    ax = fig.add_axes([0.115, 0.115, 0.660, 0.775])   # square at this canvas
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    ax.fill_between([lo, hi], [lo, hi], [lo, lo], color=S1, alpha=0.05, lw=0)
    ax.fill_between([lo, hi], [hi, hi], [lo, hi], color=S2, alpha=0.05, lw=0)
    ax.plot([lo, hi], [lo, hi], ls=(0, (5, 4)), color=INK3, lw=1.2, zorder=2)

    labels = {"cell.n.1": "biology", "cell.n.5": "prison"}
    for gold, color in zip(["cell.n.1", "cell.n.5"], (S1, S2)):
        sel = cell.gold == gold
        ax.scatter(x[sel], y[sel], s=70, color=color, edgecolor="white",
                   linewidth=1.2, zorder=4,
                   label=f"{labels[gold]}  (n={int(sel.sum())})")

    ax.set_xlabel("cosine to the biological-cell photos", fontsize=13, labelpad=6)
    ax.set_ylabel("cosine to the prison-cell photos", fontsize=13, labelpad=6)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.set_title("Each dot is one usage of “cell”", loc="left", fontsize=12.5,
                 color=INK, pad=8)
    # The two half-planes, named. Kept clear of the point clouds by hand.
    ax.text(0.50, 0.94, "assigned  prison cell", transform=ax.transAxes,
            va="top", ha="center", fontsize=12.5, color=S2, fontweight="bold")
    ax.text(0.97, 0.06, "assigned  biological cell", transform=ax.transAxes,
            va="bottom", ha="right", fontsize=12.5, color=S1, fontweight="bold")
    # The axes are square with equal ranges, so the boundary really is at 45°.
    ax.text(0.615, 0.635, "nearest anchor wins", transform=ax.transAxes,
            rotation=45, rotation_mode="anchor", va="bottom", ha="left",
            fontsize=11, color=INK3)

    ax.legend(title="gold sense", loc="upper left", bbox_to_anchor=(1.03, 1.0),
              frameon=False, fontsize=11, labelcolor=INK2, handletextpad=0.4,
              borderaxespad=0)
    fig.text(0.800, 0.63,
             f"assignment\nARI {cell.ari:.2f}\n\npermuted-photo\nnull {null.mean():.2f}",
             fontsize=11, color=INK2, va="top", linespacing=1.15)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=DPI)
    print(f"wrote {OUT} at {SIZE[0] * 25.4:.0f}x{SIZE[1] * 25.4:.0f} mm · "
          f"ARI {cell.ari:.3f} · null {null.mean():.3f}")


if __name__ == "__main__":
    main()
