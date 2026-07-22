"""Tests for hypernym-expanded visual-anchor grounding.

Uses a stable WordNet relation (``airliner`` is a direct hyponym of ``airplane``)
and derives the ImageNet WNID from the synset offset, so the test is robust to
WordNet version offset numbering.
"""
import pytest

from src.pilotlib.wordnet_utils import imagenet_ancestor_index, sense_anchors

wn = pytest.importorskip("nltk.corpus").wordnet


def _wnid(synset_name: str) -> str:
    return f"n{wn.synset(synset_name).offset():08d}"


def test_distance_and_grounding():
    airplane = wn.synset("airplane.n.01")
    airliner = wn.synset("airliner.n.01")   # direct hyponym of airplane
    wnid = _wnid("airliner.n.01")

    idx = imagenet_ancestor_index([wnid], max_dist=3)
    # The leaf grounds itself at distance 0 and its parent at distance 1.
    assert idx[airliner][wnid] == 0
    assert idx[airplane][wnid] == 1

    grounded = sense_anchors("airplane", idx, cap=12)
    assert "airplane.n.01" in grounded
    assert wnid in grounded["airplane.n.01"]


def test_max_dist_zero_only_leaf_sense():
    wnid = _wnid("airliner.n.01")
    idx0 = imagenet_ancestor_index([wnid], max_dist=0)
    # At distance 0 the parent sense is not grounded, only the leaf itself.
    assert sense_anchors("airplane", idx0, cap=12) == {}
    assert "airliner.n.01" in sense_anchors("airliner", idx0, cap=12)


def test_cap_truncates_nearest_first():
    wnids = [_wnid("airliner.n.01"), _wnid("warplane.n.01")]  # both hyponyms
    idx = imagenet_ancestor_index(wnids, max_dist=3)
    capped = sense_anchors("airplane", idx, cap=1)
    assert len(capped["airplane.n.01"]) == 1


def test_empty_index():
    assert sense_anchors("airplane", {}, cap=12) == {}
