"""Clustering evaluation metrics: ARI, V-measure, extended B-cubed."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, v_measure_score


def bcubed(gold: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """Extended B-cubed precision, recall, F1 (Bagga & Baldwin / Amigó et al.).

    For each item, precision = fraction of its cluster sharing its gold class;
    recall = fraction of its gold class sharing its cluster. Returns means.
    """
    gold = np.asarray(gold)
    pred = np.asarray(pred)
    n = len(gold)
    if n == 0:
        return 0.0, 0.0, 0.0
    same_gold = gold[:, None] == gold[None, :]
    same_pred = pred[:, None] == pred[None, :]
    correct = same_gold & same_pred
    precision = (correct.sum(axis=1) / same_pred.sum(axis=1)).mean()
    recall = (correct.sum(axis=1) / same_gold.sum(axis=1)).mean()
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def all_metrics(gold, pred) -> dict:
    gold = np.asarray(gold)
    pred = np.asarray(pred)
    p, r, f = bcubed(gold, pred)
    return {
        "ari": float(adjusted_rand_score(gold, pred)),
        "v_measure": float(v_measure_score(gold, pred)),
        "bcubed_p": p,
        "bcubed_r": r,
        "bcubed_f1": f,
    }


def paired_bootstrap(diffs: np.ndarray, n_resamples: int, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """Percentile CI of the mean of paired per-lemma differences."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n == 0:
        return {"point": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "excludes_zero": False, "n": 0}
    rng = np.random.RandomState(seed)
    means = diffs[rng.randint(0, n, size=(n_resamples, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n": n,
    }
