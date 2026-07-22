"""Extract SemEval-2010 Task 14 (WSI & Disambiguation) occurrences.

Per-lemma XML files under ``test_data/<pos>/<lemma>.<pos>.xml`` hold instances as
child elements named by instance id (e.g. ``access.n.1``); the target sentence
is wrapped in a ``<TargetSentence>`` element, with preceding/following context as
surrounding text. Unlike SemEval-2013 there is **no target-token offset**, so we
locate the lemma inside the target sentence (whole-word match, allowing simple
inflectional suffixes) and fall back to the whole target sentence if not found.

Gold senses come from the unsupervised evaluation key (``all.key``), one
OntoNotes-style sense id (``lemma.n.N``) per instance. Output is the standard
occurrence schema; the sense id is carried in ``gold_synset``.
"""
from __future__ import annotations

import argparse
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_POS_DIR = {"noun": "nouns", "verb": "verbs"}


def _read_gold(key_path: Path) -> dict[str, str]:
    """instance_id -> gold sense id (first sense if several)."""
    gold: dict[str, str] = {}
    for line in key_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3:
            gold[parts[1]] = parts[2]
    return gold


def _locate(lemma: str, sentence: str) -> tuple[int, int]:
    """Char span of the target in `sentence`; whole sentence if not found."""
    for pattern in (rf"\b{re.escape(lemma)}\w*\b", re.escape(lemma)):
        m = re.search(pattern, sentence, flags=re.IGNORECASE)
        if m:
            return m.start(), m.end()
    return 0, len(sentence)


def _parse_file(xml_path: Path):
    """Yield (instance_id, lemma, full_text, target_sentence, ts_offset)."""
    text = xml_path.read_text(encoding="utf-8", errors="replace")
    # The files are not always entity-clean; recover by escaping bare ampersands.
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        root = ET.fromstring(re.sub(r"&(?!(amp|lt|gt|quot|apos|#\d+);)", "&amp;", text))
    lemma = xml_path.name.split(".")[0]
    for inst in list(root):
        ts = inst.find("TargetSentence")
        if ts is None:
            continue
        pre = inst.text or ""
        ts_text = (ts.text or "").strip()
        post = ts.tail or ""
        full = pre + ts_text + post
        yield inst.tag, lemma, full, ts_text, len(pre)


def iter_occurrences(root: Path, gold_key: Path, pos: str):
    gold = _read_gold(gold_key)
    pos_dir = root / _POS_DIR[pos]
    if not pos_dir.is_dir():
        raise FileNotFoundError(f"no {_POS_DIR[pos]}/ directory under {root}")

    for xml_path in sorted(pos_dir.glob("*.xml")):
        for ident, lemma, full, ts_text, ts_off in _parse_file(xml_path):
            sense = gold.get(ident)
            if sense is None:
                continue
            rel_start, rel_end = _locate(lemma, ts_text)
            start, end = ts_off + rel_start, ts_off + rel_end
            yield {
                "lemma": lemma,
                "sentence_id": ident,
                "sentence": full,
                "target_start": start,
                "target_end": end,
                "target_surface": full[start:end],
                "gold_synset": sense,
            }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract SemEval-2010 Task 14 occurrences.")
    ap.add_argument("--root", required=True, help="test_data directory (has nouns/, verbs/)")
    ap.add_argument("--gold", required=True,
                    help="Unsupervised gold key (evaluation/unsup_eval/keys/all.key)")
    ap.add_argument("--pos", default="noun", choices=["noun", "verb"])
    ap.add_argument("--output", default="data/semeval2010_occurrences.parquet")
    args = ap.parse_args()

    rows = list(iter_occurrences(Path(args.root), Path(args.gold), args.pos))
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    logger.info("Wrote %d %s occurrences across %d lemmas to %s",
                len(df), args.pos, df["lemma"].nunique() if len(df) else 0, args.output)


if __name__ == "__main__":
    main()
