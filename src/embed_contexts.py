"""Encode BERT target-token contexts and CLIP context text per occurrence.

Reads the SemCor occurrences, keeps only those belonging to a selected target
and one of that target's retained senses, then produces two aligned caches:

    cache/bert_contexts.pt   {"vectors": (N,H), "meta": [...], "model": id}
    cache/clip_contexts.pt   {"vectors": (N,D), "vectors_raw": (N,D)|None,
                              "meta": [...], "model": id}

`meta` rows carry lemma / sentence_id / gold_synset / subset and are shared
row-for-row between the two caches. Occurrences whose target span is truncated
away by BERT are dropped from both, preserving alignment.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.pilotlib.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _local_window(sentence: str, start: int, end: int, left: int, right: int) -> str:
    """Return up to `left`/`right` whitespace words around the target span."""
    words, spans, cur = [], [], 0
    for tok in sentence.split(" "):
        words.append(tok)
        spans.append((cur, cur + len(tok)))
        cur += len(tok) + 1
    # Word indices overlapping the target char span.
    tgt = [i for i, (s, e) in enumerate(spans) if s < end and e > start]
    if not tgt:
        return sentence
    lo = max(0, tgt[0] - left)
    hi = min(len(words), tgt[-1] + 1 + right)
    return " ".join(words[lo:hi])


def _retained_occurrences(occ: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    subset_of, senses_of = {}, {}
    for row in targets.itertuples(index=False):
        subset_of[row.lemma] = row.subset
        senses_of[row.lemma] = set(str(row.retained_senses).split(";"))
    mask = occ.apply(
        lambda r: r["lemma"] in senses_of and r["gold_synset"] in senses_of[r["lemma"]],
        axis=1,
    )
    sel = occ[mask].copy()
    sel["subset"] = sel["lemma"].map(subset_of)
    return sel.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Encode BERT + CLIP contexts.")
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--occurrences", default="data/semcor_occurrences.parquet")
    ap.add_argument("--targets", default="data/targets.csv")
    ap.add_argument("--imagenet-index", default="data/imagenet_classes.parquet")
    ap.add_argument("--bert-output", default="cache/bert_contexts.pt")
    ap.add_argument("--clip-output", default="cache/clip_contexts.pt")
    ap.add_argument("--label-output", default="cache/label_prototypes.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    occ = pd.read_parquet(args.occurrences)
    targets = pd.read_csv(args.targets)
    sel = _retained_occurrences(occ, targets)
    logger.info("Encoding %d retained occurrences over %d lemmas",
                len(sel), sel["lemma"].nunique())

    items = [
        {"sentence": r.sentence, "target_start": int(r.target_start),
         "target_end": int(r.target_end)}
        for r in sel.itertuples(index=False)
    ]

    # --- BERT ---------------------------------------------------------------
    from src.pilotlib.embedders import BertContextEmbedder

    bert = BertContextEmbedder(cfg.models.contextual, pool_last_n_layers=4)
    bert_vecs = bert.encode(items, batch_size=int(cfg.contexts.bert_batch_size))
    valid = ~np.isnan(bert_vecs).any(axis=1)
    if (~valid).any():
        logger.warning("Dropping %d occurrences with no target subword", int((~valid).sum()))
    sel = sel[valid].reset_index(drop=True)
    bert_vecs = bert_vecs[valid]

    meta = sel[["lemma", "sentence_id", "gold_synset", "subset"]].to_dict("records")
    Path(args.bert_output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": bert_vecs, "meta": meta, "model": cfg.models.contextual},
               args.bert_output)
    logger.info("Wrote BERT contexts (%s) to %s", bert_vecs.shape, args.bert_output)

    # --- CLIP context text --------------------------------------------------
    from src.pilotlib.embedders import ClipTextEmbedder

    left, right = int(cfg.contexts.words_left), int(cfg.contexts.words_right)
    windows = [
        _local_window(r.sentence, int(r.target_start), int(r.target_end), left, right)
        for r in sel.itertuples(index=False)
    ]
    template = cfg.contexts.clip_template
    templated = [
        template.format(target=r.target_surface, context=w)
        for r, w in zip(sel.itertuples(index=False), windows)
    ]
    clip = ClipTextEmbedder(cfg.models.vision_language)
    bs = int(cfg.contexts.clip_batch_size)
    clip_vecs = clip.encode(templated, batch_size=bs)
    clip_raw = clip.encode(windows, batch_size=bs) if cfg.contexts.clip_raw_ablation else None

    torch.save(
        {"vectors": clip_vecs, "vectors_raw": clip_raw, "meta": meta,
         "model": cfg.models.vision_language},
        args.clip_output,
    )
    logger.info("Wrote CLIP contexts (%s) to %s", clip_vecs.shape, args.clip_output)

    # --- Label prototypes (control): CLIP text of "a photo of a <class>" ----
    # Computed here so all CLIP-text encoding lives in one stage. Covers every
    # anchor WNID used by a selected target, using the same encoder as t_i.
    needed: set[str] = set()
    for cell in targets["anchor_wnids"].dropna():
        needed.update(w for w in str(cell).split(";") if w)
    label_protos: dict[str, np.ndarray] = {}
    if needed:
        index = pd.read_parquet(args.imagenet_index)
        wnid_label = {
            r.wnid: (r.lemmas[0].replace("_", " ") if len(r.lemmas) else r.wnid)
            for r in index.itertuples(index=False)
        }
        wnids = [w for w in sorted(needed) if w in wnid_label]
        prompts = [f"a photo of a {wnid_label[w]}" for w in wnids]
        vecs = clip.encode(prompts, batch_size=bs) if prompts else np.empty((0, clip_vecs.shape[1]))
        label_protos = {w: vecs[i] for i, w in enumerate(wnids)}
    torch.save(
        {"prototypes": {w: torch.from_numpy(v.astype(np.float32))
                        for w, v in label_protos.items()},
         "model": cfg.models.vision_language},
        args.label_output,
    )
    logger.info("Wrote %d label prototypes to %s", len(label_protos), args.label_output)


if __name__ == "__main__":
    main()
