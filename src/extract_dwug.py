"""Extract one row per usage from a DWUG dataset into the pilot schema.

DWUG (Diachronic Word Usage Graphs, https://www.ims.uni-stuttgart.de/data/wugs)
supplies, per target word, a set of usages (sentences in context, drawn from two
time periods) and an inventory-*free* sense clustering induced by correlation
clustering over human semantic-proximity judgments. Using DWUG clusters as the
gold labels is the point of this extractor: unlike SemCor's WordNet synsets, the
senses are not read off a predefined inventory.

The output schema is identical to ``extract_semcor.py`` so every downstream
stage (select_targets -> embed_contexts -> cluster -> evaluate) is unchanged:

    lemma          bare word form (POS suffix stripped) for ImageNet lemma match
    sentence_id    DWUG usage identifier (opaque, carried as a label)
    sentence       the usage ``context`` text
    target_start   char offset of the target span in ``sentence``
    target_end     char offset (exclusive)
    target_surface the surface tokens of the target
    gold_synset    DWUG sense label ``<lemma>.cl<cluster>`` (analogous to a synset)

Design choices (see the pilot README / scoping notes):
  * Time periods are *pooled*: the ``grouping`` column is ignored so all usages
    of a word cluster together. This matches the pilot's sense-induction
    question; measuring change across periods is left to a later stage.
  * Usages assigned to DWUG's ``-1`` noise cluster (unclusterable nodes) are
    dropped. DWUG's long tail of singleton clusters is *kept* here and pruned
    later by ``data.min_occurrences_per_sense`` in select_targets, exactly as
    rare WordNet senses are pruned on the SemCor path.
  * ``pos`` filters by the DWUG directory suffix (``_nn`` noun, ``_vb`` verb).
    The pilot is nouns-only; verbs get no ImageNet anchor anyway.

Gold clusters may be used to filter/evaluate but never to build occurrence-level
features, so they are carried here only as labels.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_POS_SUFFIX = {"noun": "nn", "verb": "vb"}
_NOISE_CLUSTER = "-1"


def _read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a DWUG tab-separated table (``.csv`` files are TSV by convention).

    ``QUOTE_NONE`` is essential: DWUG contexts frequently begin with a literal
    double-quote, which the csv default would treat as a quote char and swallow
    the embedded tab delimiters, corrupting every column after ``context``.
    """
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def _strip_pos(word_dir: str) -> str:
    """``attack_nn`` -> ``attack``; leave names without a known suffix intact."""
    for suffix in _POS_SUFFIX.values():
        if word_dir.endswith(f"_{suffix}"):
            return word_dir[: -(len(suffix) + 1)]
    return word_dir


def iter_occurrences(dwug_root: Path, clustering: str, pos: str):
    """Yield occurrence dicts from a DWUG dataset directory.

    ``dwug_root`` is the extracted dataset dir (contains ``data/`` and
    ``clusters/``); ``clustering`` selects the ``clusters/<clustering>/``
    subdirectory (``opt`` is DWUG's optimized clustering).
    """
    data_dir = dwug_root / "data"
    clusters_dir = dwug_root / "clusters" / clustering
    if not data_dir.is_dir():
        raise FileNotFoundError(f"no data/ directory under {dwug_root}")
    if not clusters_dir.is_dir():
        raise FileNotFoundError(f"no clusters/{clustering}/ directory under {dwug_root}")

    want_suffix = None if pos == "all" else f"_{_POS_SUFFIX[pos]}"
    word_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())

    for word_path in word_dirs:
        word = word_path.name
        if want_suffix is not None and not word.endswith(want_suffix):
            continue
        uses_path = word_path / "uses.csv"
        cluster_path = clusters_dir / f"{word}.csv"
        if not uses_path.exists() or not cluster_path.exists():
            logger.warning("skipping %s: missing uses.csv or clusters file", word)
            continue

        clusters = {r["identifier"]: r["cluster"] for r in _read_tsv(cluster_path)}
        lemma = _strip_pos(word)

        for row in _read_tsv(uses_path):
            ident = row["identifier"]
            cluster = clusters.get(ident)
            if cluster is None or cluster == _NOISE_CLUSTER:
                continue
            context = row["context"]
            try:
                start_s, end_s = row["indexes_target_token"].split(":")
                start, end = int(start_s), int(end_s)
            except (KeyError, ValueError):
                logger.warning("bad target index for %s in %s; skipping", ident, word)
                continue
            yield {
                "lemma": lemma,
                "sentence_id": ident,
                "sentence": context,
                "target_start": start,
                "target_end": end,
                "target_surface": context[start:end],
                "gold_synset": f"{lemma}.cl{cluster}",
            }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract DWUG usages into the pilot schema.")
    ap.add_argument("--dwug-root", required=True,
                    help="Extracted DWUG dataset directory (contains data/ and clusters/)")
    ap.add_argument("--clustering", default="opt",
                    help="clusters/<name>/ subdirectory to read (default: opt)")
    ap.add_argument("--pos", default="noun", choices=["noun", "verb", "all"],
                    help="POS to keep, by DWUG dir suffix (pilot default: noun)")
    ap.add_argument("--output", default="data/dwug_occurrences.parquet")
    args = ap.parse_args()

    rows = list(iter_occurrences(Path(args.dwug_root), args.clustering, args.pos))
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    logger.info(
        "Wrote %d %s usages across %d lemmas to %s",
        len(df), args.pos, df["lemma"].nunique() if len(df) else 0, args.output,
    )


if __name__ == "__main__":
    main()
