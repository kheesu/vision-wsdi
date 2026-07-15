"""WordNet / ImageNet-WNID helpers shared across stages."""
from __future__ import annotations

from functools import lru_cache


def wnid_to_synset(wnid: str):
    """Map an ImageNet WNID (``n########``) to its WordNet synset, or None.

    ImageNet WNIDs encode ``<pos><offset>``; the numeric part is the WordNet
    byte offset. Requires ``nltk`` + the ``wordnet`` corpus.
    """
    from nltk.corpus import wordnet as wn

    if not wnid or wnid[0] != "n":
        return None
    try:
        return wn.synset_from_pos_and_offset("n", int(wnid[1:]))
    except Exception:  # noqa: BLE001 - offset may not exist in this WN version
        return None


def synset_lemmas(synset) -> list[str]:
    """Lower-cased lemma strings of a synset (underscores kept for multiword)."""
    return [lemma.name().lower() for lemma in synset.lemmas()]


@lru_cache(maxsize=1)
def _lemma_to_noun_wnids():
    """Build {lemma -> set(WNID)} restricted to noun synsets present in WN.

    This is the inventory-assisted lookup C_w: the ImageNet index (built from the
    on-disk class directories) is intersected with this at selection time.
    """
    # Not used directly for selection (we intersect with the actual on-disk
    # ImageNet classes there); kept for ad-hoc exploration.
    from nltk.corpus import wordnet as wn

    mapping: dict[str, set[str]] = {}
    for syn in wn.all_synsets("n"):
        wnid = f"n{syn.offset():08d}"
        for lemma in syn.lemmas():
            mapping.setdefault(lemma.name().lower(), set()).add(wnid)
    return mapping
