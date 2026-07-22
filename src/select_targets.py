"""Select target lemmas and assign them to evaluation subsets.

Selection criteria (from the plan):
    n_occurrences (over retained senses) >= data.min_occurrences
    n_retained_senses                    >= data.min_senses
    examples per retained sense          >= data.min_occurrences_per_sense

A word sense is *visually grounded* when an ImageNet class lies within
``data.anchor_max_hypernym_dist`` WordNet hypernym levels below it (see
``pilotlib.wordnet_utils.sense_anchors``). The visual-anchor set is the union of
those grounded senses' ImageNet classes, and subsets are keyed on the number of
*grounded senses* g_w (not the raw anchor count, which would flag a single
concrete sense with many hyponyms):
    multi_visual      g_w >= 2                    (>=2 senses with distinct anchors)
    visual_nonvisual  g_w == 1                    (one concrete sense vs other usages)
    text_only         g_w == 0                    (no anchor; kept so the text-only
                                                   baselines still have targets, e.g.
                                                   when ImageNet is absent — excluded
                                                   from the image go/no-go comparison)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.pilotlib.config import load_config
from src.pilotlib.wordnet_utils import imagenet_ancestor_index, sense_anchors

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Select target lemmas and visual subsets.")
    ap.add_argument("--occurrences", default="data/semcor_occurrences.parquet")
    ap.add_argument("--imagenet-index", default="data/imagenet_classes.parquet")
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--output", default="data/targets.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    occ = pd.read_parquet(args.occurrences)
    index = pd.read_parquet(args.imagenet_index)

    min_occ = cfg.data.min_occurrences
    min_senses = cfg.data.min_senses
    min_per_sense = cfg.data.min_occurrences_per_sense
    max_dist = cfg.data.get("anchor_max_hypernym_dist", 3)
    cap = cfg.data.get("anchor_max_per_sense", 12)

    # ancestor -> {imagenet wnid: hypernym distance}; empty when ImageNet is absent.
    ancestor_index = (
        imagenet_ancestor_index(index["wnid"].tolist(), max_dist) if len(index) else {}
    )

    rows = []
    for lemma, grp in occ.groupby("lemma"):
        sense_counts = grp["gold_synset"].value_counts()
        retained = sense_counts[sense_counts >= min_per_sense]
        if len(retained) < min_senses:
            continue
        n_occ = int(retained.sum())
        if n_occ < min_occ:
            continue

        grounded = sense_anchors(lemma, ancestor_index, cap) if ancestor_index else {}
        wnids = sorted({w for anchors in grounded.values() for w in anchors})
        n_grounded = len(grounded)
        if n_grounded >= 2:
            subset = "multi_visual"
        elif n_grounded == 1:
            subset = "visual_nonvisual"
        else:
            subset = "text_only"

        rows.append(
            {
                "lemma": lemma,
                "subset": subset,
                "n_occurrences": n_occ,
                "n_senses": len(retained),
                "gold_k": len(retained),
                "n_visual_senses": n_grounded,
                "n_visual_anchors": len(wnids),
                "anchor_wnids": ";".join(wnids),
                "anchor_senses": ";".join(sorted(grounded)),
                "retained_senses": ";".join(sorted(retained.index)),
            }
        )

    df = pd.DataFrame(rows).sort_values(["subset", "lemma"]).reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    counts = df["subset"].value_counts().to_dict() if len(df) else {}
    logger.info("Selected %d target lemmas: %s", len(df), counts)
    n_multi = counts.get("multi_visual", 0)
    if n_multi < 5:
        logger.warning(
            "Only %d multi-visual lemmas (< 5). ImageNet-1k may be too narrow "
            "for a meaningful multi-visual test; consider a larger synset "
            "collection if one is already on the box.", n_multi,
        )


if __name__ == "__main__":
    main()
