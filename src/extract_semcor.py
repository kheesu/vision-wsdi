"""Extract one row per sense-annotated noun occurrence from SemCor.

Output columns (one row per occurrence):
    lemma          WordNet lemma of the target (lower-case, underscores kept)
    sentence_id    index into semcor.tagged_sents
    sentence       whitespace-reconstructed sentence text
    target_start   char offset of the target span in `sentence`
    target_end     char offset (exclusive)
    target_surface the surface tokens of the target
    gold_synset    WordNet synset name, e.g. "dog.n.01"

Gold senses may be used to filter/evaluate but never to build occurrence-level
features, so they are carried here only as labels.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _iter_occurrences(pos: str):
    """Yield occurrence dicts from SemCor for the requested POS ('n' for noun)."""
    from nltk.corpus import semcor
    from nltk.corpus.reader.wordnet import Lemma

    sents = semcor.tagged_sents(tag="sem")
    for sent_id, chunks in enumerate(tqdm(sents, desc="semcor", unit="sent")):
        # Reconstruct the sentence and remember each chunk's char span.
        tokens: list[str] = []
        spans: list[tuple[int, int]] = []  # (start, end) per chunk in the joined text
        pieces: list[str] = []
        cursor = 0
        for chunk in chunks:
            leaves = chunk.leaves() if hasattr(chunk, "leaves") else list(chunk)
            surface = " ".join(leaves)
            start = cursor
            end = start + len(surface)
            pieces.append(surface)
            spans.append((start, end))
            tokens.extend(leaves)
            cursor = end + 1  # +1 for the joining space
        sentence = " ".join(pieces)

        for chunk, (start, end) in zip(chunks, spans):
            label = chunk.label() if hasattr(chunk, "label") else None
            if not isinstance(label, Lemma):
                continue
            synset = label.synset()
            if synset is None or synset.pos() != pos:
                continue
            yield {
                "lemma": label.name().lower(),
                "sentence_id": sent_id,
                "sentence": sentence,
                "target_start": start,
                "target_end": end,
                "target_surface": sentence[start:end],
                "gold_synset": synset.name(),
            }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract SemCor noun occurrences.")
    ap.add_argument("--output", default="data/semcor_occurrences.parquet")
    ap.add_argument("--pos", default="noun", choices=["noun"],
                    help="POS to extract (pilot is nouns only)")
    args = ap.parse_args()

    pos_code = {"noun": "n"}[args.pos]
    rows = list(_iter_occurrences(pos_code))
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    logger.info(
        "Wrote %d %s occurrences across %d lemmas to %s",
        len(df), args.pos, df["lemma"].nunique() if len(df) else 0, args.output,
    )


if __name__ == "__main__":
    main()
