"""Label-prototype assignment control (+ calibration and image/label score fusion).

The main pipeline has a `qwen+label` fusion control but no label-prototype
*assignment* — so the headline "visible anchor" result (§6.1) never controlled
for whether the image content matters or just the WordNet class identity (its
name). This closes that gap on whatever corpus is currently cached
(cache/text_contexts.pt + cache/{imagenet,label}_prototypes.pt + data/targets.csv),
CPU-only, over the multi_visual lemmas:

    img      anchor-assignment from image prototypes (reproduces the pipeline)
    lbl      same rule, class-NAME text prototypes instead of images
    img_cal / lbl_cal
             per-sense z-scored scores before argmax (future-work §9.1 probe)
    maxfuse  argmax over 0.5*img_scores + 0.5*lbl_scores (score-level fusion)

Usage: run from the repo root that holds the caches:
    .venv/bin/python experiments/agreement_gate/label_assignment.py <corpus-name>
(<corpus-name> only labels the output CSV, written next to this script.)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari

sys.path.insert(0, os.getcwd())
from src.construct_features import FeatureBank  # noqa: E402
from src.pilotlib.config import load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS = sys.argv[1] if len(sys.argv) > 1 else "cached"

cfg = load_config("configs/pilot.yaml")
fb = FeatureBank("cache/text_contexts.pt", "cache/imagenet_prototypes.pt",
                 "cache/label_prototypes.pt", "data/targets.csv",
                 pca_dim=int(cfg.contexts.pca_dimensions))


def zcal(sc):
    sd = sc.std(axis=0)
    return (sc - sc.mean(axis=0)) / np.where(sd < 1e-12, 1.0, sd)


rows = []
for lem in fb.lemmas():
    if fb.subset_of.get(lem) != "multi_visual":
        continue
    t = fb.text[fb.rows[lem]]
    gold = fb._gold_ids[lem]
    anchors = [w for w in fb.anchors.get(lem, [])
               if w in fb.image_proto and w in fb.label_proto]
    col = {w: i for i, w in enumerate(anchors)}
    sense_cols = {s: [col[w] for w in ws if w in col]
                  for s, ws in fb.grouping.get(lem, {}).items()}
    sense_cols = {s: ix for s, ix in sense_cols.items() if ix}
    if len(sense_cols) < 2:
        continue
    senses = sorted(sense_cols)

    def scores(protos):
        prof = fb._profile(t, anchors, protos)
        return np.stack([prof[:, sense_cols[s]].max(axis=1) for s in senses], axis=1)

    si, sl = scores(fb.image_proto), scores(fb.label_proto)
    rows.append(dict(lemma=lem, n=len(gold),
                     img=ari(gold, si.argmax(1)),
                     lbl=ari(gold, sl.argmax(1)),
                     img_cal=ari(gold, zcal(si).argmax(1)),
                     lbl_cal=ari(gold, zcal(sl).argmax(1)),
                     maxfuse=ari(gold, (0.5 * si + 0.5 * sl).argmax(1))))

D = pd.DataFrame(rows)
print(f"{CORPUS}: {len(D)} multi_visual lemmas")
print(D.round(3).to_string(index=False))
print("\nmeans:")
print(D.drop(columns=["lemma", "n"]).mean().round(4).to_string())

out = HERE / f"label_assignment_{CORPUS}.csv"
D.round(6).to_csv(out, index=False)
print(f"\nwrote {out}")
