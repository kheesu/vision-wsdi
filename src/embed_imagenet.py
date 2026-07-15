"""Compute normalised CLIP visual prototypes for the required ImageNet classes.

For each class c, v_c = normalize( (1/M) * sum_j f_I(x_cj) ), M = samples_per_class,
with x_cj a deterministic sample of the class's training images. Only the WNIDs
that appear as anchors of a selected target are embedded. Writes a torch dict:

    {"prototypes": {wnid: FloatTensor(D)}, "dim": D, "wnids": [...],
     "samples_per_class": M, "model": <id>}

If ImageNet is unavailable an empty-but-valid cache is written so the rest of the
pipeline runs (image-dependent systems are then skipped downstream).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from src.pilotlib.config import load_config, probe_imagenet

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_IMG_EXTS = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}


def _sample_paths(class_dir: Path, m: int, seed: int) -> list[Path]:
    files = sorted(p for p in class_dir.iterdir() if p.suffix in _IMG_EXTS)
    if len(files) <= m:
        return files
    # Deterministic: seed by (sampling_seed, wnid) so a class's sample is stable.
    rng = np.random.RandomState((seed + abs(hash(class_dir.name))) % (2**32))
    idx = rng.choice(len(files), size=m, replace=False)
    return [files[i] for i in sorted(idx)]


def _needed_wnids(targets_csv: Path) -> set[str] | None:
    if not targets_csv.exists():
        return None
    import pandas as pd

    df = pd.read_csv(targets_csv)
    wnids: set[str] = set()
    for cell in df["anchor_wnids"].dropna():
        wnids.update(w for w in str(cell).split(";") if w)
    return wnids


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed ImageNet class prototypes.")
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--targets", default="data/targets.csv")
    ap.add_argument("--output", default="cache/imagenet_prototypes.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    status = probe_imagenet(cfg.data.imagenet_root or None)

    empty = {
        "prototypes": {},
        "dim": 0,
        "wnids": [],
        "samples_per_class": cfg.images.samples_per_class,
        "model": cfg.models.vision_language,
    }
    if not status.available:
        logger.warning("ImageNet unavailable (%s); writing empty prototype cache.", status.reason)
        torch.save(empty, args.output)
        return

    needed = _needed_wnids(Path(args.targets))
    from src.pilotlib.embedders import ClipImageEmbedder

    embedder = ClipImageEmbedder(
        cfg.models.vision_language, use_fp16=bool(cfg.images.use_fp16)
    )
    m = int(cfg.images.samples_per_class)
    seed = int(cfg.images.sampling_seed)
    batch = int(cfg.images.batch_size)

    prototypes: dict[str, torch.Tensor] = {}
    class_dirs = [d for d in sorted(status.train_dir.iterdir())
                  if d.is_dir() and d.name.startswith("n")]
    for class_dir in class_dirs:
        wnid = class_dir.name
        if needed is not None and wnid not in needed:
            continue
        paths = _sample_paths(class_dir, m, seed)
        if not paths:
            continue
        feats = embedder.encode_paths(paths, batch_size=batch)  # (m, D) unit vectors
        if feats.shape[0] == 0:
            continue
        proto = feats.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-12)
        prototypes[wnid] = torch.from_numpy(proto.astype(np.float32))
        logger.info("prototype %s from %d images", wnid, feats.shape[0])

    dim = len(next(iter(prototypes.values()))) if prototypes else 0
    torch.save(
        {
            "prototypes": prototypes,
            "dim": dim,
            "wnids": sorted(prototypes),
            "samples_per_class": m,
            "model": cfg.models.vision_language,
        },
        args.output,
    )
    logger.info("Wrote %d prototypes (dim=%d) to %s", len(prototypes), dim, args.output)


if __name__ == "__main__":
    main()
