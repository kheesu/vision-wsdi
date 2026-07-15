"""Build per-lemma feature matrices for every system in the comparison.

Systems (see the plan's controls table):
    bert                 h~_i           global PCA_64 of BERT target-token vectors
    clip-context         t_i            CLIP context-text embedding
    bert+image           z^λ            [h~ ; λ·zscore(a_img)]  (image prototypes)
    bert+label           z^λ            [h~ ; λ·zscore(a_lbl)]  (label prototypes)
    bert+shuffled-image  z^λ            image prototypes permuted across classes
    image-profile-only   a_img          diagnostic

The anchor profile a_i[c] = cos(t_i, v_c). To keep image and label comparable
they use the *same* candidate classes: the anchors that have both an image and a
label prototype. Everything except the shuffle is seed-independent and computed
once; the shuffle permutation is drawn per seed.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

FUSION_SYSTEMS = ("bert+image", "bert+label", "bert+shuffled-image")


def _zscore(a: np.ndarray) -> np.ndarray:
    mu = a.mean(axis=0, keepdims=True)
    sd = a.std(axis=0, keepdims=True)
    return (a - mu) / np.where(sd < 1e-12, 1.0, sd)


def _l2(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.clip(n, 1e-12, None)


class FeatureBank:
    def __init__(self, bert_path, clip_path, image_proto_path, label_proto_path,
                 targets_csv, pca_dim: int):
        bert = torch.load(bert_path, weights_only=False)
        clip = torch.load(clip_path, weights_only=False)
        self.meta = pd.DataFrame(bert["meta"])
        self.bert = np.asarray(bert["vectors"], dtype=np.float32)
        self.clip = np.asarray(clip["vectors"], dtype=np.float32)

        img = torch.load(image_proto_path, weights_only=False)
        lbl = torch.load(label_proto_path, weights_only=False)
        self.image_proto = {w: v.numpy() for w, v in img.get("prototypes", {}).items()}
        self.label_proto = {w: v.numpy() for w, v in lbl.get("prototypes", {}).items()}

        tgt = pd.read_csv(targets_csv)
        self.anchors = {
            r.lemma: [w for w in str(r.anchor_wnids).split(";") if w]
            for r in tgt.itertuples(index=False)
        }
        self.subset_of = dict(zip(tgt["lemma"], tgt["subset"]))
        self.gold_k = dict(zip(tgt["lemma"], tgt["gold_k"]))

        # Global PCA over all retained BERT occurrence vectors.
        n_comp = min(pca_dim, self.bert.shape[0], self.bert.shape[1])
        self.pca = PCA(n_components=n_comp, random_state=0).fit(self.bert)
        self.bert_pca = self.pca.transform(self.bert).astype(np.float32)

        # Gold label ids per lemma, and row indices per lemma.
        self.rows = {lem: idx.to_numpy() for lem, idx in
                     self.meta.groupby("lemma").groups.items()}
        self._gold_ids = {}
        for lem, ridx in self.rows.items():
            senses = self.meta.loc[ridx, "gold_synset"].to_numpy()
            _, inv = np.unique(senses, return_inverse=True)
            self._gold_ids[lem] = inv

        # Ordered list of image-prototype WNIDs for the shuffle permutation.
        self._proto_wnids = sorted(self.image_proto)

    def lemmas(self) -> list[str]:
        return [lem for lem in self.rows if lem in self.anchors]

    def _profile(self, t: np.ndarray, wnids: list[str], protos: dict) -> np.ndarray:
        """cos(t_i, v_c) for each c; t and protos are unit vectors -> dot."""
        cols = [t @ protos[w] for w in wnids]
        return np.stack(cols, axis=1) if cols else np.empty((t.shape[0], 0), dtype=np.float32)

    def build(self, lemma: str, lambdas, seed: int) -> dict:
        ridx = self.rows[lemma]
        h = self.bert_pca[ridx]
        t = self.clip[ridx]
        gold = self._gold_ids[lemma]

        out = {
            "subset": self.subset_of.get(lemma, "text_only"),
            "gold_k": int(self.gold_k.get(lemma, len(np.unique(gold)))),
            "gold": gold,
            "n": len(ridx),
            "systems": {"bert": h, "clip-context": t},
        }

        # Candidate classes shared by image + label prototypes.
        anchors = self.anchors.get(lemma, [])
        img_anchors = [w for w in anchors if w in self.image_proto and w in self.label_proto]
        if img_anchors:
            a_img = self._profile(t, img_anchors, self.image_proto)
            a_lbl = self._profile(t, img_anchors, self.label_proto)
            out["systems"]["image-profile-only"] = a_img

            # Per-seed shuffle: permute prototypes across the global class list,
            # then read this lemma's anchors from the permuted mapping.
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(self._proto_wnids))
            shuffled = {self._proto_wnids[i]: self.image_proto[self._proto_wnids[perm[i]]]
                        for i in range(len(self._proto_wnids))}
            a_shuf = self._profile(t, [w for w in img_anchors], shuffled)

            zc_img, zc_lbl, zc_shuf = _zscore(a_img), _zscore(a_lbl), _zscore(a_shuf)
            for lam in lambdas:
                out["systems"].setdefault("bert+image", {})[lam] = _l2(
                    np.hstack([h, lam * zc_img]))
                out["systems"].setdefault("bert+label", {})[lam] = _l2(
                    np.hstack([h, lam * zc_lbl]))
                out["systems"].setdefault("bert+shuffled-image", {})[lam] = _l2(
                    np.hstack([h, lam * zc_shuf]))
        return out
