"""Figure for Result 1 (Q1): nearest-visible-anchor assignment vs. permuted null.

Recomputes the REPORT.md §6 numbers from each run's bootstrap.json per_lemma
table — macro-ARI over `multi_visual` words under the ImageNet-21k inventory,
with the paired bootstrap (resample words, keep each word's assignment-minus-
null difference paired) — then renders a single dot plot: the paired difference
per corpus with its 95% CI, zero line for reference. Sized and set for a
poster, so the type is large enough to read from a distance.

Usage:
    .venv/bin/python -m experiments.figures.figure_q1 [--results results] [--out figures/figure_q1_benchmarks.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.pilotlib.metrics import paired_bootstrap

ROOT = Path(__file__).resolve().parents[2]
CORPORA = [
    ("semcor", "SemCor"),
    ("dwug_en", "DWUG EN"),
    ("semeval2013", "SemEval-2013"),
    ("semeval2010", "SemEval-2010"),
]
RUN_SUFFIX = "_21k_oracle_k"  # assignment is K-free; oracle/unknown identical
SEED = 13
N_RESAMPLES = 10_000

# palette / chrome (light mode; background transparent — marker rings assume
# the figure lands on a light page)
SURFACE = "#ffffff"
SERIES_BLUE = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# type scale, poster-sized (no headline: the poster section supplies it)
FS_CAPTION = 18
FS_AXIS_TITLE = 20
FS_CORPUS = 21
FS_TICK = 18
FS_VALUE = 19


def corpus_stats(run_dir: Path) -> dict:
    per_lemma = json.loads((run_dir / "bootstrap.json").read_text())["per_lemma"]
    rows = [r for r in per_lemma.values() if r["subset"] == "multi_visual"]
    assign = np.array([r["anchor_assignment"] for r in rows])
    null = np.array([r["assignment_null"] for r in rows])
    boot = paired_bootstrap(assign - null, N_RESAMPLES, seed=SEED)
    return {"assign": assign.mean(), "null": null.mean(), **boot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "figures/figure_q1_benchmarks.png")
    args = ap.parse_args()

    stats = {
        label: corpus_stats(args.results / f"{key}{RUN_SUFFIX}")
        for key, label in CORPORA
    }
    for label, s in stats.items():
        print(
            f"{label:13s} n={s['n']:3d}  assign {s['assign']:+.3f}  null {s['null']:+.3f}"
            f"  delta {s['point']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]"
            f"  excludes 0: {s['excludes_zero']}"
        )

    order = sorted(stats, key=lambda c: stats[c]["point"], reverse=True)
    y = np.arange(len(order))[::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.4), facecolor="none")
    ax.set_facecolor("none")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.grid(axis="x", color=GRID, linewidth=1.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelsize=FS_TICK, colors=MUTED)
    ax.set_ylim(-0.75, len(order) - 0.25)

    ax.axvline(0, color=BASELINE, lw=1.4, zorder=1)
    for yi, label in zip(y, order):
        s = stats[label]
        ax.errorbar(
            s["point"], yi,
            xerr=[[s["point"] - s["ci_low"]], [s["ci_high"] - s["point"]]],
            fmt="o", ms=14, mfc=SERIES_BLUE, mec=SURFACE, mew=3,
            ecolor=SERIES_BLUE, elinewidth=3.2, capsize=7, capthick=3.2, zorder=3,
        )
        ax.annotate(
            f"{s['point']:+.3f}",
            (s["point"], yi), xytext=(0, 17), textcoords="offset points",
            ha="center", fontsize=FS_VALUE, color=INK_2,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{label}  (n={stats[label]['n']})" for label in order],
        color=INK_2, fontsize=FS_CORPUS,
    )
    ax.set_xlim(-0.10, 0.37)
    ax.set_xticks([-0.1, 0.0, 0.1, 0.2, 0.3])
    ax.set_title(
        "Δ macro-ARI  (assignment − permuted null)",
        loc="left", fontsize=FS_AXIS_TITLE, color=INK, pad=16,
    )

    # anchored to the axes, not the figure, so it left-aligns with the title
    # (a figure-relative x drifts once bbox_inches="tight" crops in the long
    # corpus labels)
    ax.text(
        0.0, 1.17,
        "Multi-visual words, ImageNet-21k inventory\n"
        "95% CI from a paired bootstrap over words",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=FS_CAPTION, color=INK_2, linespacing=1.5,
    )

    args.out.parent.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", pad_inches=0.3,
                transparent=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
