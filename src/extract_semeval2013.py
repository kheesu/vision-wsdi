"""Extract SemEval-2013 Task 13 (WSI for graded/non-graded senses) occurrences.

The test data ships per-lemma XML files under ``contexts/xml-format/`` and gold
keys under ``keys/gold/``. Each ``<instance>`` carries the target token's
character offsets (``tokenStart``/``tokenEnd``) in the instance text, so the
target span is exact. Gold senses are WordNet-3.1 sense keys; instances may be
*graded* (several weighted senses). For hard clustering we keep only the
**single-sense** instances by default (drop graded ones), matching how the
pilot drops ambiguous data elsewhere; ``--include-graded`` instead keeps every
instance and takes its highest-weighted sense.

Output is the standard occurrence schema (see ``extract_semcor.py``); the gold
sense key goes in ``gold_synset`` as an opaque per-lemma label.
"""
from __future__ import annotations

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_POS_CODE = {"noun": "n", "verb": "v", "adj": "j"}


def _read_gold(key_path: Path) -> dict[str, list[tuple[str, float]]]:
    """instance_id -> [(sense_key, weight), ...] from a SemEval-2013 gold key."""
    gold: dict[str, list[tuple[str, float]]] = {}
    for line in key_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        _lemma_pos, instance_id, *sense_fields = parts
        senses = []
        for field in sense_fields:
            sense, _, weight = field.partition("/")
            senses.append((sense, float(weight) if weight else 1.0))
        gold[instance_id] = senses
    return gold


def iter_occurrences(root: Path, pos: str, include_graded: bool):
    """Yield occurrence dicts for the requested POS."""
    code = _POS_CODE[pos]
    gold = _read_gold(root / "keys" / "gold" / "all.key")
    xml_dir = root / "contexts" / "xml-format"
    if not xml_dir.is_dir():
        raise FileNotFoundError(f"no contexts/xml-format/ under {root}")

    for xml_path in sorted(xml_dir.glob(f"*.{code}.xml")):
        tree = ET.parse(xml_path)
        for inst in tree.getroot().findall("instance"):
            ident = inst.get("id")
            senses = gold.get(ident)
            if not senses:
                continue
            if len(senses) > 1 and not include_graded:
                continue  # graded instance dropped in single-sense mode
            # highest-weighted sense (a no-op for single-sense instances)
            gold_sense = max(senses, key=lambda sw: sw[1])[0]
            text = inst.text or ""
            start, end = int(inst.get("tokenStart")), int(inst.get("tokenEnd"))
            yield {
                "lemma": inst.get("lemma"),
                "sentence_id": ident,
                "sentence": text,
                "target_start": start,
                "target_end": end,
                "target_surface": text[start:end],
                "gold_synset": gold_sense,
            }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract SemEval-2013 Task 13 occurrences.")
    ap.add_argument("--root", required=True,
                    help="Extracted SemEval-2013-Task-13-test-data directory")
    ap.add_argument("--pos", default="noun", choices=["noun", "verb", "adj"])
    ap.add_argument("--include-graded", action="store_true",
                    help="keep multi-sense instances (take max-weight sense)")
    ap.add_argument("--output", default="data/semeval2013_occurrences.parquet")
    args = ap.parse_args()

    rows = list(iter_occurrences(Path(args.root), args.pos, args.include_graded))
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    logger.info("Wrote %d %s occurrences across %d lemmas to %s",
                len(df), args.pos, df["lemma"].nunique() if len(df) else 0, args.output)


if __name__ == "__main__":
    main()
