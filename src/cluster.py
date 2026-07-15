"""Spherical k-means clustering per lemma for every system.

Two modes:
    oracle   K = gold number of senses (controlled representation test)
    unknown  K chosen in [k_min, min(k_max, floor(n/k_denom))] by cosine silhouette

Writes ``<output>/assignments.parquet`` (one row per method/lambda/seed/lemma,
with gold + predicted label arrays) and ``<output>/run.json`` (mode + config).
Downstream ``evaluate.py`` turns these into metrics.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.construct_features import FUSION_SYSTEMS, FeatureBank
from src.pilotlib.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _l2(a: np.ndarray) -> np.ndarray:
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)


def _spherical_kmeans(x: np.ndarray, k: int, seed: int, n_init: int) -> np.ndarray:
    """KMeans on L2-normalised rows == cosine k-means."""
    xn = _l2(x)
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    return km.fit_predict(xn)


def _choose_k(x: np.ndarray, k_lo: int, k_hi: int, seed: int, n_init: int) -> tuple[int, np.ndarray]:
    xn = _l2(x)
    best_k, best_labels, best_score = k_lo, None, -np.inf
    for k in range(k_lo, k_hi + 1):
        labels = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(xn)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(xn, labels, metric="cosine")
        if score > best_score:
            best_k, best_labels, best_score = k, labels, score
    if best_labels is None:  # degenerate: everything collapses
        best_labels = np.zeros(len(x), dtype=int)
    return best_k, best_labels


def _iter_variants(systems: dict, lambdas):
    """Yield (method, lambda_or_None, feature_matrix)."""
    for name, val in systems.items():
        if name in FUSION_SYSTEMS:
            for lam in lambdas:
                yield name, float(lam), val[lam]
        else:
            yield name, None, val


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster occurrences per lemma.")
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--output", default="results/oracle_k")
    ap.add_argument("--mode", choices=["oracle", "unknown"], default=None)
    ap.add_argument("--bert", default="cache/bert_contexts.pt")
    ap.add_argument("--clip", default="cache/clip_contexts.pt")
    ap.add_argument("--image-prototypes", default="cache/imagenet_prototypes.pt")
    ap.add_argument("--label-prototypes", default="cache/label_prototypes.pt")
    ap.add_argument("--targets", default="data/targets.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or ("oracle" if cfg.clustering.oracle_k else "unknown")
    lambdas = list(cfg.fusion.lambdas)
    seeds = list(cfg.clustering.seeds)
    n_init = int(cfg.clustering.n_init)

    bank = FeatureBank(
        args.bert, args.clip, args.image_prototypes, args.label_prototypes,
        args.targets, pca_dim=int(cfg.contexts.pca_dimensions),
    )
    lemmas = bank.lemmas()
    logger.info("Clustering %d lemmas in %s mode over %d seeds", len(lemmas), mode, len(seeds))

    rows = []
    for seed in seeds:
        for lemma in lemmas:
            feats = bank.build(lemma, lambdas, seed)
            n = feats["n"]
            gold = feats["gold"]
            gold_k = feats["gold_k"]
            k_hi = min(cfg.clustering.unknown_k_max, n // cfg.clustering.unknown_k_denom)
            k_lo = cfg.clustering.unknown_k_min
            for method, lam, x in _iter_variants(feats["systems"], lambdas):
                if x.shape[1] == 0:
                    continue
                if mode == "oracle":
                    pred_k = min(gold_k, n)
                    pred = _spherical_kmeans(x, pred_k, seed, n_init)
                else:
                    if k_hi < k_lo:  # too few occurrences to search
                        pred_k, pred = min(gold_k, n), _spherical_kmeans(x, min(gold_k, n), seed, n_init)
                    else:
                        pred_k, pred = _choose_k(x, k_lo, k_hi, seed, n_init)
                rows.append(
                    {
                        "method": method,
                        "lambda": lam if lam is not None else -1.0,
                        "seed": seed,
                        "lemma": lemma,
                        "subset": feats["subset"],
                        "n_occurrences": n,
                        "gold_k": gold_k,
                        "predicted_k": int(pred_k),
                        "gold": gold.tolist(),
                        "pred": [int(p) for p in pred],
                    }
                )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_dir / "assignments.parquet", index=False)
    (out_dir / "run.json").write_text(
        json.dumps({"mode": mode, "lambdas": lambdas, "seeds": seeds,
                    "n_lemmas": len(lemmas)}, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %d clustering rows to %s", len(rows), out_dir / "assignments.parquet")


if __name__ == "__main__":
    main()
