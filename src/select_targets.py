"""Select target lemmas and assign them to evaluation subsets.

Selection criteria (from the plan):
    n_occurrences (over retained senses) >= data.min_occurrences
    n_retained_senses                    >= data.min_senses
    examples per retained sense          >= data.min_occurrences_per_sense

Visual-anchor set C_w = {ImageNet class c : w is a WordNet lemma of c}. Subsets:
    multi_visual      |C_w| >= 2                 (distinct senses may have distinct anchors)
    visual_nonvisual  |C_w| == 1, >= 2 senses    (one concrete sense vs other usages)
    text_only         |C_w| == 0                 (no anchor; kept so the text-only
                                                   baselines still have targets, e.g.
                                                   when ImageNet is absent — excluded
                                                   from the image go/no-go comparison)
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.pilotlib.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _lemma_to_wnids(index: pd.DataFrame) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for row in index.itertuples(index=False):
        for lemma in row.lemmas:
            mapping[lemma].append(row.wnid)
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description="Select SemCor target lemmas.")
    ap.add_argument("--occurrences", default="data/semcor_occurrences.parquet")
    ap.add_argument("--imagenet-index", default="data/imagenet_classes.parquet")
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--output", default="data/targets.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    occ = pd.read_parquet(args.occurrences)
    index = pd.read_parquet(args.imagenet_index)
    lemma2wnids = _lemma_to_wnids(index) if len(index) else {}

    min_occ = cfg.data.min_occurrences
    min_senses = cfg.data.min_senses
    min_per_sense = cfg.data.min_occurrences_per_sense

    rows = []
    for lemma, grp in occ.groupby("lemma"):
        sense_counts = grp["gold_synset"].value_counts()
        retained = sense_counts[sense_counts >= min_per_sense]
        if len(retained) < min_senses:
            continue
        n_occ = int(retained.sum())
        if n_occ < min_occ:
            continue

        wnids = sorted(set(lemma2wnids.get(lemma, [])))
        if len(wnids) >= 2:
            subset = "multi_visual"
        elif len(wnids) == 1:
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
                "n_visual_anchors": len(wnids),
                "anchor_wnids": ";".join(wnids),
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
