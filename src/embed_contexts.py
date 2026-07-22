"""Encode word-in-context text with Qwen3-VL-Embedding.

Reads the occurrences, keeps only those belonging to a selected target and one
of that target's retained senses, and embeds a local context window per
occurrence (plain text, the model's default instruction). Writes:

    cache/text_contexts.pt   {"vectors": (N,D), "meta": [...], "model": id}
    cache/label_prototypes.pt {"prototypes": {wnid: Tensor(D)}, "model": id}

`meta` rows carry lemma / sentence_id / gold_synset / subset. The one text
embedding is reused downstream as both the clustering base and the anchor query,
so there is no separate context encoding. Label prototypes are the text
embedding of each anchor class's name — the control that isolates whether the
class *name* (rather than its image) explains any effect.
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
    ap = argparse.ArgumentParser(description="Encode word-in-context text with Qwen3-VL.")
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--occurrences", default="data/semcor_occurrences.parquet")
    ap.add_argument("--targets", default="data/targets.csv")
    ap.add_argument("--imagenet-index", default="data/imagenet_classes.parquet")
    ap.add_argument("--text-output", default="cache/text_contexts.pt")
    ap.add_argument("--label-output", default="cache/label_prototypes.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    occ = pd.read_parquet(args.occurrences)
    targets = pd.read_csv(args.targets)
    sel = _retained_occurrences(occ, targets)
    logger.info("Encoding %d retained occurrences over %d lemmas",
                len(sel), sel["lemma"].nunique())

    left, right = int(cfg.contexts.words_left), int(cfg.contexts.words_right)
    windows = [
        _local_window(r.sentence, int(r.target_start), int(r.target_end), left, right)
        for r in sel.itertuples(index=False)
    ]

    from src.pilotlib.embedders import QwenEmbedder

    embedder = QwenEmbedder(
        cfg.models.embedding,
        dtype=cfg.embedding.get("dtype", "bfloat16"),
        prompt=cfg.embedding.get("prompt", None),
    )
    text_vecs = embedder.encode_texts(windows, batch_size=int(cfg.embedding.text_batch_size))

    meta = sel[["lemma", "sentence_id", "gold_synset", "subset"]].to_dict("records")
    Path(args.text_output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": text_vecs, "meta": meta, "model": cfg.models.embedding},
               args.text_output)
    logger.info("Wrote text embeddings (%s) to %s", text_vecs.shape, args.text_output)

    # --- Label prototypes (control): text embedding of each anchor class name --
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
        names = [wnid_label[w] for w in wnids]
        vecs = (embedder.encode_texts(names, batch_size=int(cfg.embedding.text_batch_size))
                if names else np.empty((0, text_vecs.shape[1]), dtype=np.float32))
        label_protos = {w: vecs[i] for i, w in enumerate(wnids)}
    torch.save(
        {"prototypes": {w: torch.from_numpy(v.astype(np.float32))
                        for w, v in label_protos.items()},
         "model": cfg.models.embedding},
        args.label_output,
    )
    logger.info("Wrote %d label prototypes to %s", len(label_protos), args.label_output)


if __name__ == "__main__":
    main()
