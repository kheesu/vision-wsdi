"""Index the on-disk ImageNet class directories.

For every ``n########`` directory found under the ImageNet root (or its
``train/`` subdir) record the WNID, the WordNet synset it maps to, that synset's
lemmas, and how many image files are present. When ImageNet is not available an
empty (but correctly-typed) index is written so downstream stages still run with
an empty visual-anchor set.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.pilotlib.config import probe_imagenet
from src.pilotlib.wordnet_utils import synset_lemmas, wnid_to_synset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_IMG_EXTS = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}
_COLUMNS = ["wnid", "synset", "lemmas", "n_images"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Index ImageNet class directories.")
    ap.add_argument("--root", default="", help="ImageNet root")
    ap.add_argument("--output", default="data/imagenet_classes.parquet")
    args = ap.parse_args()

    status = probe_imagenet(args.root or None)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if not status.available:
        logger.warning("ImageNet unavailable (%s); writing empty index.", status.reason)
        pd.DataFrame(columns=_COLUMNS).to_parquet(args.output, index=False)
        return

    rows = []
    for class_dir in sorted(status.train_dir.iterdir()):
        if not class_dir.is_dir() or not class_dir.name.startswith("n"):
            continue
        synset = wnid_to_synset(class_dir.name)
        if synset is None:
            logger.warning("WNID %s has no WordNet synset; skipping.", class_dir.name)
            continue
        n_images = sum(1 for p in class_dir.iterdir() if p.suffix in _IMG_EXTS)
        rows.append(
            {
                "wnid": class_dir.name,
                "synset": synset.name(),
                "lemmas": synset_lemmas(synset),
                "n_images": n_images,
            }
        )

    df = pd.DataFrame(rows, columns=_COLUMNS)
    df.to_parquet(args.output, index=False)
    logger.info("Indexed %d ImageNet classes to %s", len(df), args.output)


if __name__ == "__main__":
    main()
