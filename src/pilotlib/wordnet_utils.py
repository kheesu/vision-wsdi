"""WordNet / ImageNet-WNID helpers shared across stages."""
from __future__ import annotations


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


def imagenet_ancestor_index(wnids, max_dist: int):
    """Map every WordNet synset to the ImageNet classes it *visually grounds*.

    ImageNet-1k classes are specific leaf synsets (``airliner``, ``soccer_ball``)
    while the polysemous words we study carry higher-level senses (``airplane``,
    ``ball``). A leaf class ``c`` is taken to ground a sense ``s`` when ``c`` lies
    within ``max_dist`` hypernym levels *below* ``s`` (``s == c`` is distance 0).
    This is the link that a bare surface-lemma match misses.

    Returns ``{ancestor_synset: {imagenet_wnid: min_hypernym_distance}}`` over the
    given on-disk ImageNet ``wnids``. A small ``max_dist`` keeps groundings tight;
    a large one drags in over-general senses (``matter`` -> half of ImageNet).
    """
    index: dict = {}
    for wnid in wnids:
        leaf = wnid_to_synset(wnid)
        if leaf is None:
            continue
        # BFS upward from the leaf, recording the min distance to each ancestor.
        best: dict = {}
        frontier = {leaf: 0}
        while frontier:
            nxt: dict = {}
            for node, dist in frontier.items():
                if node in best and best[node] <= dist:
                    continue
                best[node] = dist
                if dist < max_dist:
                    for parent in node.hypernyms() + node.instance_hypernyms():
                        if parent not in best or best[parent] > dist + 1:
                            nxt[parent] = min(nxt.get(parent, dist + 1), dist + 1)
            frontier = nxt
        for node, dist in best.items():
            index.setdefault(node, {})[wnid] = dist
    return index


def sense_anchors(lemma: str, ancestor_index: dict, cap: int) -> dict[str, list[str]]:
    """Visual anchors per WordNet noun sense of ``lemma``.

    For each noun sense that grounds at least one ImageNet class in
    ``ancestor_index``, return the anchoring WNIDs (nearest first, capped at
    ``cap``). ``{sense_name: [wnid, ...]}``; senses with no grounding are omitted,
    so ``len(result)`` is the count of visually-grounded senses.
    """
    from nltk.corpus import wordnet as wn

    grounded: dict[str, list[str]] = {}
    for sense in wn.synsets(lemma, pos="n"):
        hits = ancestor_index.get(sense)
        if not hits:
            continue
        ordered = sorted(hits, key=lambda w: (hits[w], w))  # nearest first, stable
        grounded[sense.name()] = ordered[:cap] if cap else ordered
    return grounded
