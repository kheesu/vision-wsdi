"""Unit tests for the evaluation metrics."""
import numpy as np

from src.pilotlib.metrics import bcubed, paired_bootstrap


def test_bcubed_perfect():
    gold = np.array([0, 0, 1, 1])
    pred = np.array([5, 5, 9, 9])  # label values are irrelevant, partition matches
    p, r, f = bcubed(gold, pred)
    assert p == 1.0 and r == 1.0 and f == 1.0


def test_bcubed_all_one_cluster():
    # Everything in one cluster: recall perfect, precision = fraction same-class.
    gold = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 0, 0])
    p, r, f = bcubed(gold, pred)
    assert r == 1.0
    assert np.isclose(p, 0.5)  # each item: 2 of 4 cluster-mates share its class


def test_bcubed_singletons():
    # Every item its own cluster: precision perfect, recall = 1/class_size.
    gold = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 2, 3])
    p, r, f = bcubed(gold, pred)
    assert p == 1.0
    assert np.isclose(r, 0.5)


def test_paired_bootstrap_positive_excludes_zero():
    diffs = np.full(20, 0.1)
    res = paired_bootstrap(diffs, n_resamples=2000, seed=0)
    assert res["excludes_zero"] is True
    assert np.isclose(res["point"], 0.1)


def test_paired_bootstrap_centered_includes_zero():
    rng = np.random.RandomState(0)
    diffs = rng.normal(0, 1, size=50)
    res = paired_bootstrap(diffs, n_resamples=5000, seed=0)
    assert res["excludes_zero"] is False


def test_paired_bootstrap_empty():
    res = paired_bootstrap(np.array([]), n_resamples=100)
    assert res["n"] == 0 and res["excludes_zero"] is False
