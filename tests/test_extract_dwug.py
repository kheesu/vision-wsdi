"""Unit test for the DWUG extractor against a tiny synthetic dataset dir.

No network/download: a minimal ``data/<word>/uses.csv`` + ``clusters/opt`` tree
is built on disk in the DWUG TSV layout, including the awkward cases the parser
must survive (a context beginning with a literal double-quote, and a ``-1``
noise cluster).
"""
from pathlib import Path

import pandas as pd

from src.extract_dwug import iter_occurrences

USES_COLS = [
    "lemma", "pos", "date", "grouping", "identifier", "description", "context",
    "indexes_target_token", "indexes_target_sentence", "context_tokenized",
    "indexes_target_token_tokenized", "indexes_target_sentence_tokenized",
    "context_lemmatized", "context_pos",
]


def _row(identifier, context, span, date="1850", grouping="1"):
    start, end = span
    r = {c: " " for c in USES_COLS}
    r.update(lemma="ball_nn", pos="nn1", date=date, grouping=grouping,
             identifier=identifier, context=context,
             indexes_target_token=f"{start}:{end}")
    return r


def _write_tsv(path: Path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(cols)]
    lines += ["\t".join(str(r[c]) for c in cols) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_dwug(root: Path):
    word = "ball_nn"
    uses = [
        _row("u1", "He kicked the ball hard.", (14, 18), grouping="1"),
        # Context opening with a literal quote — must NOT be treated as csv quoting.
        _row("u2", '"The ball was grand," she said.', (5, 9), grouping="2"),
        _row("u3", "She went to the ball in a gown.", (16, 20), grouping="2"),
        _row("u4", "A noisy ball rolled away.", (8, 12), grouping="1"),
    ]
    _write_tsv(root / "data" / word / "uses.csv", uses, USES_COLS)
    clusters = [
        {"identifier": "u1", "cluster": "0"},   # sport sense
        {"identifier": "u2", "cluster": "0"},
        {"identifier": "u3", "cluster": "1"},    # dance sense
        {"identifier": "u4", "cluster": "-1"},   # noise -> dropped
    ]
    _write_tsv(root / "clusters" / "opt" / f"{word}.csv", clusters,
               ["identifier", "cluster"])
    # A verb dir that the noun POS filter must skip.
    _write_tsv(root / "data" / "run_vb" / "uses.csv",
               [_row("v1", "They run fast.", (5, 8))], USES_COLS)
    _write_tsv(root / "clusters" / "opt" / "run_vb.csv",
               [{"identifier": "v1", "cluster": "0"}], ["identifier", "cluster"])


def test_iter_occurrences(tmp_path):
    _build_dwug(tmp_path)
    df = pd.DataFrame(iter_occurrences(tmp_path, clustering="opt", pos="noun"))

    # u4 dropped (noise), run_vb skipped (verb) -> 3 noun rows, one lemma.
    assert len(df) == 3
    assert set(df["lemma"]) == {"ball"}
    assert "run" not in set(df["lemma"])

    # POS suffix stripped; cluster carried as the gold label.
    assert set(df["gold_synset"]) == {"ball.cl0", "ball.cl1"}

    # Target spans are exact even for the quote-leading context (regression).
    for r in df.itertuples():
        assert r.sentence[r.target_start:r.target_end] == r.target_surface
    assert df.loc[df["sentence_id"] == "u2", "target_surface"].item() == "ball"


def test_pos_all_keeps_verbs(tmp_path):
    _build_dwug(tmp_path)
    df = pd.DataFrame(iter_occurrences(tmp_path, clustering="opt", pos="all"))
    assert "run" in set(df["lemma"])
