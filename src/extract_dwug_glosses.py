"""Extract DWUG-EN cluster glosses (definitions) as sense anchors.

Reads the `wugs_with_definitions` repo (ltgoslo), which pairs every DWUG
correlation-cluster with a machine-generated definition. Emits one row per
(lemma, cluster) so each *induced sense* can be anchored by its gloss text —
an inventory-free alternative to the WordNet->ImageNet visual anchors that needs
no WordNet and aligns 1:1 with the gold clusters being scored.

Placeholder glosses ("Too few examples to generate a proper definition!") are
dropped; they only occur for tiny clusters that the pilot's per-sense threshold
already discards.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_PLACEHOLDER = "Too few examples"


def _strip_pos(word_dir: str) -> str:
    for suf in ("nn", "vb"):
        if word_dir.endswith(f"_{suf}"):
            return word_dir[: -(len(suf) + 1)]
    return word_dir


def iter_glosses(gloss_root: Path, pos: str):
    """Yield {lemma, cluster, gloss} from wug_labels/english/english_labels/."""
    base = gloss_root / "wug_labels" / "english" / "english_labels"
    if not base.is_dir():
        raise FileNotFoundError(f"no english_labels under {gloss_root}")
    want = None if pos == "all" else pos[0]  # 'n'->nn, 'v'->vb suffix
    for word_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        name = word_dir.name
        if want is not None and not name.endswith(f"_{ {'n':'nn','v':'vb'}[want] }"):
            continue
        tsv = word_dir / "cluster_gloss.tsv"
        if not tsv.exists():
            continue
        lemma = _strip_pos(name)
        with tsv.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                gloss = (row.get("gloss") or "").strip()
                if not gloss or gloss.startswith(_PLACEHOLDER):
                    continue
                yield {"lemma": lemma, "cluster": int(row["cluster"]), "gloss": gloss}


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract DWUG-EN cluster glosses.")
    ap.add_argument("--gloss-root", required=True, help="clone of ltgoslo/wugs_with_definitions")
    ap.add_argument("--pos", default="noun", choices=["noun", "verb", "all"])
    ap.add_argument("--output", default="data/dwug_glosses.csv")
    args = ap.parse_args()
    pos = {"noun": "n", "verb": "v", "all": "all"}[args.pos]
    rows = list(iter_glosses(Path(args.gloss_root), pos))
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Wrote %d glosses across %d lemmas to %s",
                len(df), df["lemma"].nunique() if len(df) else 0, args.output)


if __name__ == "__main__":
    main()
