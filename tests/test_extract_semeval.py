"""Unit tests for the SemEval-2013 and SemEval-2010 WSI extractors.

Synthetic dataset dirs are built on disk in each benchmark's real layout; no
download. They exercise the tricky bits: SemEval-2013's single-sense filtering
and explicit token offsets, and SemEval-2010's target-word location inside
``<TargetSentence>`` with surrounding context.
"""
from pathlib import Path

import pandas as pd

from src.extract_semeval2010 import iter_occurrences as iter_2010
from src.extract_semeval2013 import iter_occurrences as iter_2013


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------- SemEval-2013 ---------------------------------- #
def _build_2013(root: Path):
    ctx = '<?xml version="1.0" encoding="UTF-8"?>\n<instances lemma="crane" partOfSpeech="n">\n'
    # "The tall crane lifted." -> "crane" at 9:14
    ctx += ('  <instance id="crane.n.1" lemma="crane" partOfSpeech="n" token="crane"'
            ' tokenStart="9" tokenEnd="14">The tall crane lifted.</instance>\n')
    # "A crane waded." -> "crane" at 2:7
    ctx += ('  <instance id="crane.n.2" lemma="crane" partOfSpeech="n" token="crane"'
            ' tokenStart="2" tokenEnd="7">A crane waded.</instance>\n')
    ctx += ('  <instance id="crane.n.3" lemma="crane" partOfSpeech="n" token="crane"'
            ' tokenStart="2" tokenEnd="7">A crane thing.</instance>\n</instances>\n')
    _write(root / "contexts" / "xml-format" / "crane.n.xml", ctx)
    # A verb file the noun POS filter must skip.
    _write(root / "contexts" / "xml-format" / "run.v.xml",
           '<?xml version="1.0" encoding="UTF-8"?>\n<instances lemma="run" partOfSpeech="v">\n'
           '  <instance id="run.v.1" lemma="run" partOfSpeech="v" token="run"'
           ' tokenStart="2" tokenEnd="5">I run fast.</instance>\n</instances>\n')
    _write(root / "keys" / "gold" / "all.key",
           "crane.n crane.n.1 crane%1:06:00::/1\n"
           "crane.n crane.n.2 crane%1:05:00::/1\n"
           "crane.n crane.n.3 crane%1:06:00::/1 crane%1:05:00::/1\n"   # graded
           "run.v run.v.1 run%2:38:00::/1\n")


def test_semeval2013_single_sense(tmp_path):
    _build_2013(tmp_path)
    df = pd.DataFrame(iter_2013(tmp_path, pos="noun", include_graded=False))
    assert set(df["sentence_id"]) == {"crane.n.1", "crane.n.2"}   # graded n.3 dropped
    assert set(df["lemma"]) == {"crane"}                          # verb skipped
    assert set(df["gold_synset"]) == {"crane%1:06:00::", "crane%1:05:00::"}
    for r in df.itertuples():
        assert r.sentence[r.target_start:r.target_end] == r.target_surface == "crane"


def test_semeval2013_include_graded(tmp_path):
    _build_2013(tmp_path)
    df = pd.DataFrame(iter_2013(tmp_path, pos="noun", include_graded=True))
    assert "crane.n.3" in set(df["sentence_id"])   # kept, max-weight sense


# --------------------------- SemEval-2010 ---------------------------------- #
def _build_2010(root: Path):
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           "<crane.n.test>"
           "<crane.n.1>Intro context.  <TargetSentence>The crane lifted a load.  "
           "</TargetSentence>Trailing bits.</crane.n.1>"
           "<crane.n.2>Other lead.  <TargetSentence>A crane flew over.  "
           "</TargetSentence>End.</crane.n.2>"
           "</crane.n.test>\n")
    _write(root / "test_data" / "nouns" / "crane.n.xml", xml)
    _write(root / "gold.key",
           "crane.n crane.n.1 crane.n.5\ncrane.n crane.n.2 crane.n.2\n")


def test_semeval2010_locates_target(tmp_path):
    _build_2010(tmp_path)
    df = pd.DataFrame(iter_2010(tmp_path / "test_data", tmp_path / "gold.key", pos="noun"))
    assert len(df) == 2
    assert set(df["gold_synset"]) == {"crane.n.5", "crane.n.2"}
    for r in df.itertuples():
        # span is exact within the assembled full context, surface is the target
        assert r.sentence[r.target_start:r.target_end] == r.target_surface == "crane"
        # full context includes the surrounding sentences, not just the target one
        assert "Intro context" in r.sentence or "Other lead" in r.sentence
