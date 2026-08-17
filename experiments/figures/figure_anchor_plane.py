"""The showcase figure: usages land on visible anchors.

Sized for one column of a three-column A0 poster: 200 x 250 mm portrait, 300 dpi.
Type sizes are physical points on the printed poster, so the panel furniture is
deliberately sparse -- prose belongs in the poster body, not in here.

  A  The anchor plane (the hero). Coordinates are the *actual* quantities the
     method computes -- max_c cos(t_i, v_c) over each grounded sense's ImageNet
     classes -- so the diagonal y = x IS the decision rule, with no
     dimensionality reduction and no distortion between the numbers and the
     picture. For `cell` the plotted 2-way boundary reproduces the full 4-way
     argmax over the candidate senses for all 69 usages (asserted below).
  B  Text embeddings alone. Spherical k-means on the clustering base recovers
     the same split, but the groups are anonymous and K has to be chosen.
  C  The per-usage arithmetic behind panel A, for the two usages panel A quotes:
     one cosine per candidate sense, argmax wins. This is what makes the method
     inductive -- a usage that was never part of any clustering gets a named
     sense from one comparison -- and it shows the two candidates (electric
     cell, mobile phone) that never win, which is why panel A can be 2-D.

Why not a joint PCA of usages + anchors: text and image embeddings sit in
separate cones of the shared space (the modality gap -- for `cell`, PC1 of the
joint set separates modality, usages at +0.12..+0.31 vs anchors at
-0.66..-0.41). A joint scatter would put the anchors in a far corner and the
"nearest anchor" reading of 2-D distance would be false. The anchor plane keeps
the axes literal instead.

Layout is hand-placed; `Placer` asserts that no photo inset or callout box
covers a data point, so the figure fails loudly rather than hiding a usage.

Run: .venv/bin/python -m experiments.figures.figure_anchor_plane
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from src.cluster import _spherical_kmeans

ROOT = Path(__file__).resolve().parents[2]
IMAGENET = Path(os.environ.get("IMAGENET_DIR", "/cldata/ImageNet-merged21k/train"))
RUN = ROOT / "results/semeval2010_21k_oracle_k"
OUT = ROOT / "figures/figure_anchor_plane.png"
SEEDS = (13, 17, 19, 23, 29, 31, 37, 41, 43, 47)  # configs/pilot.yaml

MM = 1 / 25.4
SIZE = (200 * MM, 250 * MM)   # one column of a three-column A0 poster
DPI = 300

# --- design tokens (dataviz reference palette, light surface) ---------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8983"
GRID = "#e6e5e1"
S1 = "#2a78d6"  # categorical slot 1
S2 = "#eb6834"  # categorical slot 2
NEUTRAL = "#767570"

# Physical point sizes on the printed poster. At 200 mm width a line of 8 pt
# text spans ~140 characters, so the copy budget is tight: keep every string
# below the widths noted at its call site.
FS = {"eyebrow": 7.5, "title": 17, "lede": 8.5, "panel": 11, "blurb": 7.6,
      "axis": 8.5, "tick": 7.5, "note": 8, "photo": 9, "photo_sub": 6.8,
      "quote": 7, "foot": 6.8}

mpl.rcParams.update({
    "font.family": ["Noto Sans", "DejaVu Sans"],  # fallback carries → and −
    "font.size": FS["note"],
    "axes.unicode_minus": False,   # Noto Sans here lacks U+2212
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "xtick.color": INK3,
    "ytick.color": INK3,
    "xtick.labelsize": FS["tick"],
    "ytick.labelsize": FS["tick"],
    "text.color": INK,
    "axes.linewidth": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "savefig.facecolor": SURFACE,
})
MONO = {"family": "Noto Sans Mono"}


# --- data ------------------------------------------------------------------
def load():
    text = torch.load(ROOT / "cache/text_contexts.pt", weights_only=False)
    meta = pd.DataFrame(text["meta"])
    X = np.asarray(text["vectors"], dtype=np.float32)
    proto = {k: v.numpy() for k, v in torch.load(
        ROOT / "cache/imagenet_prototypes.pt", weights_only=False)["prototypes"].items()}
    # The run scores only anchors that have both an image and a label prototype
    # (so the image/label control is comparable); keep the same candidate set
    # here or the plotted argmax would not be the run's argmax.
    label = torch.load(ROOT / "cache/label_prototypes.pt", weights_only=False)
    proto = {k: v for k, v in proto.items() if k in label["prototypes"]}
    targets = pd.read_csv(ROOT / "data/targets.csv").set_index("lemma")
    classes = pd.read_parquet(ROOT / "data/imagenet_classes.parquet").set_index("wnid")
    occ = pd.read_parquet(ROOT / "data/semeval2010_occurrences.parquet")
    return meta, X, proto, targets, classes, occ


class Word:
    """Everything a panel needs for one lemma."""

    def __init__(self, lemma, meta, X, proto, targets):
        self.lemma = lemma
        self.rows = meta.index[meta.lemma == lemma].to_numpy()
        self.t = X[self.rows]
        self.gold = meta.loc[self.rows, "gold_synset"].to_numpy()
        self.sent_ids = meta.loc[self.rows, "sentence_id"].to_numpy()
        groups = {}
        for part in str(targets.loc[lemma, "anchor_grouping"]).split("|"):
            sense, wnids = part.split("=")
            kept = [w for w in wnids.split(",") if w in proto]
            if kept:
                groups[sense] = kept
        self.groups = groups
        self.senses = sorted(groups)
        # a_i[c] = cos(t_i, v_c); sense score = max over that sense's classes.
        # Class counts differ per sense, so keep per-sense blocks in a list.
        self.sim = [self.t @ np.stack([proto[w] for w in groups[s]]).T
                    for s in self.senses]
        self.score = np.stack([s.max(axis=1) for s in self.sim], axis=1)
        self.pred = self.score.argmax(axis=1)
        self.ari = adjusted_rand_score(self.gold, self.pred)

    def shuffled_ari(self, proto, seeds):
        """The run's null: permute which prototype belongs to which class."""
        wnids = sorted(proto)
        out = []
        for seed in seeds:
            perm = np.random.RandomState(seed).permutation(len(wnids))
            sh = {wnids[i]: proto[wnids[perm[i]]] for i in range(len(wnids))}
            score = np.stack([(self.t @ np.stack([sh[c] for c in self.groups[s]]).T
                               ).max(axis=1) for s in self.senses], axis=1)
            out.append(adjusted_rand_score(self.gold, score.argmax(axis=1)))
        return np.array(out)

    def top_classes(self, sense_i):
        """Anchor classes of a sense, ordered by how often they win the argmax."""
        sel = self.pred == sense_i
        wnids = self.groups[self.senses[sense_i]]
        order = []
        if sel.sum():
            wins = pd.Series([wnids[i] for i in self.sim[sense_i][sel].argmax(axis=1)])
            order = list(wins.value_counts().index)
        return order + [w for w in wnids if w not in order]


# --- photos ----------------------------------------------------------------
def montage(wnids, side=260, seed=13):
    """2x2 tile of real ImageNet photos, cycling the given classes."""
    rng = random.Random(seed)
    tiles, i = [], 0
    while len(tiles) < 4 and i < 16:
        wnid = wnids[i % len(wnids)]
        i += 1
        files = sorted((IMAGENET / wnid).glob("*.JPEG"))
        if not files:
            continue
        try:
            im = Image.open(files[rng.randrange(len(files))]).convert("RGB")
        except OSError:
            continue
        w, h = im.size
        c = min(w, h)
        im = im.crop(((w - c) // 2, (h - c) // 2, (w - c) // 2 + c, (h - c) // 2 + c))
        tiles.append(np.asarray(im.resize((side, side), Image.LANCZOS)))
    grid = np.full((2 * side, 2 * side, 3), 252, dtype=np.uint8)
    for k, tile in enumerate(tiles[:4]):
        r, c = divmod(k, 2)
        grid[r * side:(r + 1) * side, c * side:(c + 1) * side] = tile
    return grid


def quote(occ, sent_id, width=34):
    """One-line excerpt that always keeps the target token in view.

    Words are peeled off either side until the line fits, so a narrow callout
    still shows the word in its context rather than a stray sentence opening.
    """
    row = occ[occ.sentence_id == sent_id].iloc[0]
    s, a, b = row.sentence, int(row.target_start), int(row.target_end)
    target = f"[{s[a:b]}]"
    left = " ".join(s[max(0, a - 90):a].split())[-60:].split()
    right = " ".join(s[b:b + 90].split())[:60].split()
    while True:
        text = f"…{' '.join(left)} {target} {' '.join(right)}…"
        if len(text) <= width or not (left or right):
            return text
        if len(left) >= len(right) and left:
            left = left[1:]
        elif right:
            right = right[:-1]


# --- layout guard ----------------------------------------------------------
class Placer:
    """Places boxes in axes-fraction space and refuses to cover data points."""

    def __init__(self, ax, xy):
        lo_x, hi_x = ax.get_xlim()
        lo_y, hi_y = ax.get_ylim()
        self.fx = (xy[:, 0] - lo_x) / (hi_x - lo_x)
        self.fy = (xy[:, 1] - lo_y) / (hi_y - lo_y)

    def check(self, rect, what, pad=0.015):
        x0, y0, w, h = rect
        hit = ((self.fx > x0 - pad) & (self.fx < x0 + w + pad)
               & (self.fy > y0 - pad) & (self.fy < y0 + h + pad))
        if hit.any():
            raise AssertionError(
                f"{what} at {rect} would cover {int(hit.sum())} data point(s) "
                f"near ({self.fx[hit][0]:.2f}, {self.fy[hit][0]:.2f})")
        return rect


def anchor_inset(ax, placer, grid, rect, color, title, sub):
    """Photo montage pinned inside the half-plane it governs."""
    ins = ax.inset_axes(placer.check(rect, f"photos '{title}'"))
    ins.imshow(grid)
    ins.set_xticks([])
    ins.set_yticks([])
    for spine in ins.spines.values():
        spine.set(color=color, linewidth=2.0)
    ins.set_title(title, color=color, fontsize=FS["photo"], fontweight="bold", pad=3)
    ins.text(0.5, -0.05, sub, transform=ins.transAxes, ha="center", va="top",
             fontsize=FS["photo_sub"], color=INK2, **MONO)


# Baseline-to-baseline distance is the font's own line height times matplotlib's
# `linespacing`; for Noto Sans that factor is ~1.46, and the block layout below
# only works if the title offset is computed with it rather than guessed.
LINE_H = 1.46
LS = 1.15


def head(ax, letter, title, blurb="", x=0.0):
    """Title above blurb in *point* offsets, so short panels don't collide.

    `x` is in axes fractions and may be negative, to keep a panel whose tick
    labels eat into the left margin aligned with the rest of the page.
    """
    lines = blurb.count("\n") + 1 if blurb else 0
    if blurb:
        ax.annotate(blurb, (x, 1), xycoords="axes fraction", xytext=(0, 5),
                    textcoords="offset points", va="bottom", fontsize=FS["blurb"],
                    color=INK2, linespacing=LS)
    ax.annotate(f"{letter} · {title}", (x, 1), xycoords="axes fraction",
                xytext=(0, 7 + lines * FS["blurb"] * LINE_H * LS),
                textcoords="offset points",
                va="bottom", fontsize=FS["panel"], fontweight="bold", color=INK)


def dress(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# --- panels ----------------------------------------------------------------
def panel_plane(ax, word, xi, yi, axis_names, gold_names, lims, photos,
                callouts=(), occ=None, mark_wrong=False, legend_rect=None):
    """The anchor plane: position = the cosines the method actually uses."""
    x, y = word.score[:, xi], word.score[:, yi]
    lo, hi = lims
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.fill_between([lo, hi], [lo, hi], [lo, lo], color=S1, alpha=0.05, lw=0)
    ax.fill_between([lo, hi], [hi, hi], [lo, hi], color=S2, alpha=0.05, lw=0)
    ax.plot([lo, hi], [lo, hi], ls=(0, (5, 4)), color=INK3, lw=1.1, zorder=2)

    placer = Placer(ax, np.column_stack([x, y]))
    golds = list(dict.fromkeys(word.gold))
    for g, color in zip(golds, (S1, S2)):
        sel = word.gold == g
        ax.scatter(x[sel], y[sel], s=42, color=color, edgecolor=SURFACE,
                   linewidth=1.1, alpha=0.95, zorder=4,
                   label=f"gold: {gold_names.get(g, g)}  (n={int(sel.sum())})")
    if mark_wrong:
        expected = {golds[0]: xi, golds[1]: yi}
        wrong = np.array([p != expected[g] for p, g in zip(word.pred, word.gold)])
        if wrong.any():
            ax.scatter(x[wrong], y[wrong], s=150, facecolor="none",
                       edgecolor=INK2, linewidth=1.0, zorder=5)
            ax.annotate(f"{int(wrong.sum())} of {len(x)} on the wrong side",
                        (x[wrong][0], y[wrong][0]), textcoords="offset points",
                        xytext=(8, 34), fontsize=FS["blurb"], color=INK2, ha="left",
                        arrowprops=dict(arrowstyle="-", color=INK3, lw=0.7,
                                        shrinkA=5, shrinkB=3))

    ax.set_xlabel(f"cosine to nearest {axis_names[0]} photo", fontsize=FS["axis"])
    ax.set_ylabel(f"cosine to nearest {axis_names[1]} photo", fontsize=FS["axis"])
    dress(ax)
    for (grid, title, sub, rect), color in zip(photos, (S1, S2)):
        anchor_inset(ax, placer, grid, rect, color, title, sub)

    for cal in callouts:
        i = cal["i"]
        placer.check(cal["rect"], "callout")
        ax.annotate(quote(occ, word.sent_ids[i]), (x[i], y[i]), xycoords="data",
                    xytext=(cal["rect"][0], cal["rect"][1] + cal["rect"][3] / 2),
                    textcoords="axes fraction", fontsize=FS["quote"], color=INK,
                    ha="left", va="center", zorder=6, **MONO,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff",
                              edgecolor=cal["color"], linewidth=1.0),
                    arrowprops=dict(arrowstyle="-|>", color=cal["color"], lw=1.1,
                                    shrinkA=4, shrinkB=6,
                                    connectionstyle=cal.get("cs", "arc3,rad=0.15")))
    if legend_rect:
        ax.legend(loc="upper right", frameon=False, fontsize=FS["note"],
                  labelcolor=INK2, handletextpad=0.4, borderaxespad=0,
                  bbox_to_anchor=legend_rect, bbox_transform=ax.transAxes)


def panel_text(ax, word, k):
    """The usual WSI view: accurate groups, no names."""
    h = PCA(n_components=64, random_state=0).fit_transform(word.t)
    lab = _spherical_kmeans(h, k, seed=13, n_init=20)
    Y = PCA(n_components=2, random_state=0).fit_transform(word.t)
    ari = adjusted_rand_score(word.gold, lab)
    for j, marker in zip(range(k), ("o", "^")):
        sel = lab == j
        ax.scatter(Y[sel, 0], Y[sel, 1], s=24, marker=marker,
                   facecolor="none" if j else NEUTRAL, edgecolor=NEUTRAL,
                   linewidth=1.1, alpha=0.9)
        ax.annotate(f"group {j + 1}  ?", (Y[sel, 0].mean(), Y[sel, 1].max()),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=FS["blurb"], color=INK2, fontweight="bold")
    ax.set_xlabel("PC 1", fontsize=FS["axis"], labelpad=1)
    ax.set_ylabel("PC 2", fontsize=FS["axis"], labelpad=1)
    ax.margins(0.20)
    ax.set_xticks([])
    ax.set_yticks([])
    dress(ax)
    return ari


def panel_bars(ax, word, usages, sense_labels, occ, height=0.32):
    """One cosine per candidate sense, argmax wins -- the whole labelling rule."""
    # Rows ordered by the strongest score either usage gives them: the two
    # candidates that never win end up at the bottom, where they belong.
    rank = np.argsort(-np.max([word.score[i] for i, _, _ in usages], axis=0))
    ypos = np.arange(len(rank))[::-1].astype(float)
    handles = []
    for k, (i, color, tag) in enumerate(usages):
        off = (k - (len(usages) - 1) / 2) * (height + 0.05)
        scores = word.score[i][rank]
        win = int(scores.argmax())
        bars = ax.barh(ypos - off, scores, height=height, color=color, zorder=3)
        for j, bar in enumerate(bars):        # hue = which usage, solid = winner
            bar.set_alpha(1.0 if j == win else 0.28)
        handles.append((bars[win], tag))
        for j, v in enumerate(scores):
            if j == win:
                ax.annotate(f"{v:.2f}   argmax → assigned", (v, ypos[j] - off),
                            textcoords="offset points", xytext=(5, 0), va="center",
                            fontsize=FS["note"], color=color, fontweight="bold")
            else:
                ax.annotate(f"{v:.2f}", (v, ypos[j] - off),
                            textcoords="offset points", xytext=(4, 0),
                            va="center", fontsize=FS["blurb"], color=INK3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([sense_labels[word.senses[j]] for j in rank],
                       fontsize=FS["axis"], color=INK2)
    ax.set_xlim(0, max(word.score[i].max() for i, _, _ in usages) * 1.72)
    ax.set_xlabel("cosine between the usage and the sense's nearest photo",
                  fontsize=FS["axis"], labelpad=2)
    dress(ax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend([h for h, _ in handles], [t for _, t in handles],
              loc="lower right", frameon=False, labelcolor=INK2,
              handlelength=0.8, handletextpad=0.4, borderaxespad=0.1,
              prop={"family": "Noto Sans Mono", "size": FS["quote"]})


# --- assembly --------------------------------------------------------------
def main():
    meta, X, proto, targets, classes, occ = load()
    metrics = pd.read_csv(RUN / "metrics.csv")

    def run_ari(lemma, method="anchor-assignment"):
        """The run's own per-seed ARIs (assignment is seed-free; its null is not)."""
        sel = metrics[(metrics.lemma == lemma) & (metrics.method == method)]
        return sel.ari.to_numpy()

    cell = Word("cell", meta, X, proto, targets)
    bio, prison = cell.senses.index("cell.n.02"), cell.senses.index("cell.n.07")
    # The plotted 2-way boundary must reproduce the real 4-way argmax.
    two = np.where(cell.score[:, bio] > cell.score[:, prison], bio, prison)
    assert (two == cell.pred).all(), "2-D panel would misstate the decision"
    assert abs(cell.ari - run_ari("cell").mean()) < 5e-3, "cache no longer matches the run"
    null = run_ari("cell", "anchor-assignment-shuffled")
    # Sanity check only: the prototypes were re-sampled after the run finished,
    # so a recompute lands near the run's null rather than exactly on it.
    assert abs(cell.shuffled_ari(proto, SEEDS).mean() - null.mean()) < 0.15

    def class_name(wnid, lemma):
        """A name that distinguishes the class from the target word itself."""
        lemmas = [str(x) for x in classes.loc[wnid, "lemmas"]]
        other = [x for x in lemmas if x.lower() != lemma.lower()]
        return other[0] if other else str(classes.loc[wnid, "synset"])

    def photo(word, sense_i, title, rect, seed=13, k=4):
        """A 2x2 sample of the sense's anchor classes, most-often-winning first.

        `k` caps how many distinct classes the tiles are drawn from and `seed`
        picks which photos: both only choose *which* samples are shown, and are
        set per anchor so the printed tiles read clearly at poster size.
        """
        wnids = word.top_classes(sense_i)
        sub = class_name(wnids[0], word.lemma)
        if len(wnids) > 1:
            sub += f" + {len(wnids) - 1} more"
        return montage(wnids[:k], seed=seed), title, sub, rect

    fig = plt.figure(figsize=SIZE, dpi=DPI)
    #                     left  bottom  width  height   (figure fractions)
    ax_a = fig.add_axes([0.105, 0.353, 0.519, 0.415])   # 4.09 x 4.09 in, square
    ax_b = fig.add_axes([0.680, 0.620, 0.190, 0.152])   # 1.50 x 1.50 in
    ax_c = fig.add_axes([0.225, 0.105, 0.760, 0.128])   # 5.98 x 1.26 in

    fig.text(0.105, 0.988, "SEEING WORD SENSES", fontsize=FS["eyebrow"], color=S1,
             fontweight="bold", va="top", **MONO)
    fig.text(0.105, 0.978, "A sense is whichever picture\nthe usage looks like",
             fontsize=FS["title"], fontweight="bold", color=INK, va="top",
             linespacing=1.0)
    fig.text(0.105, 0.903,     # <= 112 characters per line at 8.5 pt
             "One encoder embeds sentences and photographs in the same space, so a usage is scored against candidate\n"
             "pictures directly: a_i[c] = cos(t_i, v_c) over the ImageNet classes that WordNet grounds the word's senses in.\n"
             "The usage takes the sense whose photos it scores highest on — no clustering, no K, and the sense has a name.",
             fontsize=FS["lede"], color=INK2, va="top", linespacing=LS)

    order = np.argsort(cell.score[:, bio] - cell.score[:, prison])
    head(ax_a, "A", "Visual anchors: the sense, named",
         "69 usages of “cell” at the two cosines the method\n"   # <= 52 chars/line
         "computes. The dashed diagonal is the whole rule.")
    panel_plane(
        ax_a, cell, bio, prison,
        axis_names=("biological-cell", "prison-cell"),
        gold_names={"cell.n.1": "biology", "cell.n.5": "prison"},
        lims=(0.155, 0.655),
        photos=[photo(cell, bio, "biological cell", (0.700, 0.045, 0.275, 0.275)),
                photo(cell, prison, "prison cell", (0.360, 0.700, 0.270, 0.270), seed=4, k=2)],
        callouts=[
            {"i": int(order[0]), "rect": (0.020, 0.900, 0.32, 0.07), "color": S2,
             "cs": "arc3,rad=-0.2"},
            {"i": int(order[-1]), "rect": (0.020, 0.035, 0.34, 0.07), "color": S1,
             "cs": "arc3,rad=0.22"},
        ],
        occ=occ, mark_wrong=True, legend_rect=(1.0, 0.995),
    )

    head(ax_b, "B", "Text alone")
    ari_text = panel_text(ax_b, cell, k=2)

    fig.text(0.680, 0.578,     # <= 34 characters per line at 7.6 pt
             f"k-means on the same usages splits\nthem correctly (ARI {ari_text:.2f}), but the\n"
             "groups carry no name and no\npictures, and K had to be given.",
             fontsize=FS["blurb"], color=INK2, va="top", linespacing=LS)
    fig.text(0.680, 0.487, "nearest-anchor assignment", fontsize=FS["blurb"],
             color=INK3, va="top")
    fig.text(0.680, 0.469, f"ARI {cell.ari:.2f}", fontsize=15, color=INK,
             fontweight="bold", va="top")
    fig.text(0.680, 0.423,
             f"permuted-photo null {null.mean():.2f}\n"
             f"({len(null)} permutations,\n{null.min():.2f} to {null.max():.2f}; see the note below)",
             fontsize=FS["blurb"], color=INK2, va="top", linespacing=LS)

    head(ax_c, "C", "One comparison per candidate — the whole rule",
         "the two quoted usages again: every candidate sense of “cell” scored, argmax wins.\n"
         "A usage that was in no clustering is labelled exactly the same way.",
         x=(0.105 - 0.225) / 0.760)
    panel_bars(
        ax_c, cell,
        usages=[(int(order[0]), S2, "prison usage (from A)"),
                (int(order[-1]), S1, "biology usage (from A)")],
        sense_labels={"cell.n.02": "biological cell", "cell.n.07": "prison cell",
                      "cell.n.03": "electric cell",
                      "cellular_telephone.n.01": "mobile phone"},
        occ=occ,
    )

    fig.text(0.105, 0.058,     # <= 145 characters per line at 6.8 pt
             "SemEval-2010 Task 14 · Qwen3-VL-Embedding-8B, one encoder for text and images · anchors = ImageNet-21k leaf classes within 3 WordNet hypernym levels\n"
             "of a sense of the word, each the mean of 32 photos (tiles are samples). Gold senses colour the dots and score the ARI, never as input. The null permutes\n"
             "which photos belong to which class: on this word a permuted anchor sometimes splits the usages too, so what the real anchors add is the grounding.",
             fontsize=FS["foot"], color=INK3, va="top", linespacing=1.3)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=DPI)
    print(f"wrote {OUT} at {SIZE[0] * 25.4:.0f}x{SIZE[1] * 25.4:.0f} mm, {DPI} dpi · "
          f"cell ARI {cell.ari:.3f} (run {run_ari('cell').mean():.3f}) · "
          f"null {null.mean():.3f} [{null.min():.2f}, {null.max():.2f}] · "
          f"text-only ARI {ari_text:.3f}")


if __name__ == "__main__":
    main()
