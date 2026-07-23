"""End-to-end test of cluster -> evaluate -> report on synthetic caches.

The pipeline now uses a single Qwen text embedding as both the clustering base
and the anchor query, so the fixture is built so that the sense signal lives in a
*low-variance* subspace that the global PCA drops (leaving the `qwen` base only
weakly separable), while the image prototypes point exactly at that subspace so
the explicit anchor profile recovers it. The label anchor is a weakened version
and the shuffled anchor is uninformative, so the pipeline should produce a GO
with ARI(qwen+image) > ARI(qwen+label) > ARI(qwen) ~ ARI(shuffled).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = 8          # high-variance noise dims (PCA keeps these -> weak base)
SENSE = 8         # low-variance sense subspace (PCA drops; anchors recover it)
D = BASE + SENSE
PCA_DIM = 8
BASE_STD = 0.6
# Occurrence noise is large relative to the label's weak sense contrast but small
# relative to the exact image prototype's, so image resolves senses and label
# only partly does.
SENSE_AMP = 0.4
SENSE_NOISE = 0.3


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _build_fixtures(tmp: Path, seed: int = 0):
    rng = np.random.RandomState(seed)
    lemmas = ["alpha", "beta", "gamma", "delta"]
    per_sense = 30

    text_rows, meta, targ_rows = [], [], []
    img_proto, lbl_proto = {}, {}

    for li, lemma in enumerate(lemmas):
        w0, w1 = f"n{li:08d}", f"n{li + 100:08d}"
        # Two orthonormal directions in the sense subspace.
        q, _ = np.linalg.qr(rng.normal(size=(SENSE, 2)))
        s0, s1 = q[:, 0], q[:, 1]

        def _full(sense_vec):
            return _unit(np.concatenate([np.zeros(BASE), sense_vec]))

        v0, v1 = _full(s0), _full(s1)
        img_proto[w0], img_proto[w1] = v0, v1
        # Label prototypes: cross-contaminated, so they only weakly distinguish
        # the two senses (weaker control than the exact image prototypes).
        lbl_proto[w0] = _unit(0.65 * v0 + 0.35 * v1)
        lbl_proto[w1] = _unit(0.65 * v1 + 0.35 * v0)

        for sense_idx, s_dir in [(0, s0), (1, s1)]:
            syn = f"{lemma}.n.0{sense_idx + 1}"
            for _ in range(per_sense):
                base = BASE_STD * rng.normal(size=BASE)          # pure noise base
                sense = SENSE_AMP * s_dir + SENSE_NOISE * rng.normal(size=SENSE)
                text_rows.append(_unit(np.concatenate([base, sense])))
                meta.append({"lemma": lemma, "sentence_id": len(meta),
                             "gold_synset": syn, "subset": "multi_visual"})
        targ_rows.append({
            "lemma": lemma, "subset": "multi_visual", "n_occurrences": 2 * per_sense,
            "n_senses": 2, "gold_k": 2, "n_visual_senses": 2, "n_visual_anchors": 2,
            "anchor_wnids": f"{w0};{w1}", "anchor_senses": f"{lemma}.n.01;{lemma}.n.02",
            "anchor_grouping": f"{lemma}.n.01={w0}|{lemma}.n.02={w1}",
            "retained_senses": f"{lemma}.n.01;{lemma}.n.02",
        })

    text = np.asarray(text_rows, dtype=np.float32)

    (tmp / "cache").mkdir(parents=True, exist_ok=True)
    (tmp / "data").mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": text, "meta": meta, "model": "fake-qwen"}, tmp / "cache/text.pt")
    torch.save({"prototypes": {k: torch.from_numpy(v.astype(np.float32))
                               for k, v in img_proto.items()}, "dim": D},
               tmp / "cache/img.pt")
    torch.save({"prototypes": {k: torch.from_numpy(v.astype(np.float32))
                               for k, v in lbl_proto.items()}}, tmp / "cache/lbl.pt")
    pd.DataFrame(targ_rows).to_csv(tmp / "data/targets.csv", index=False)

    cfg = {
        "seed": 13,
        "fusion": {"lambdas": [0.5, 1.0, 2.0]},
        "contexts": {"pca_dimensions": PCA_DIM},
        "clustering": {"n_init": 5, "seeds": [13, 17, 19], "oracle_k": True,
                       "unknown_k_min": 2, "unknown_k_max": 4, "unknown_k_denom": 5},
        "evaluation": {"bootstrap_resamples": 2000},
    }
    (tmp / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    return tmp


def _run(mod, *args, cwd):
    subprocess.run([sys.executable, "-m", mod, *args], cwd=cwd, check=True)


def test_end_to_end_go(tmp_path):
    tmp = _build_fixtures(tmp_path)
    run_dir = tmp / "results/oracle_k"
    common = ["--config", str(tmp / "cfg.yaml")]

    _run("src.cluster", *common, "--mode", "oracle", "--output", str(run_dir),
         "--text", str(tmp / "cache/text.pt"),
         "--image-prototypes", str(tmp / "cache/img.pt"),
         "--label-prototypes", str(tmp / "cache/lbl.pt"),
         "--targets", str(tmp / "data/targets.csv"), cwd=ROOT)
    _run("src.evaluate", "--run", str(run_dir), *common, cwd=ROOT)
    _run("src.report", "--run", str(run_dir), cwd=ROOT)

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["n_visual_lemmas"] == 4
    assert summary["macro_ari_qwen_image"] > summary["macro_ari_qwen"]
    assert summary["macro_ari_qwen_image"] > summary["macro_ari_qwen_label"]
    # Shuffled control must not match the real image gain.
    assert summary["macro_ari_qwen_image"] > summary["macro_ari_qwen_shuffled"]

    bootstrap = json.loads((run_dir / "bootstrap.json").read_text())
    assert bootstrap["delta_image_vs_qwen"]["point"] > 0
    # Nearest-anchor assignment recovers senses above its permuted null.
    assert summary["macro_ari_anchor_assignment"] > summary["macro_ari_assignment_null"]
    assert bootstrap["delta_assignment_signal"]["point"] > 0
    assert (run_dir / "report.md").exists()
    assert (run_dir / "metrics.csv").exists()


def test_unknown_k_runs(tmp_path):
    tmp = _build_fixtures(tmp_path)
    run_dir = tmp / "results/unknown_k"
    common = ["--config", str(tmp / "cfg.yaml")]
    _run("src.cluster", *common, "--mode", "unknown", "--output", str(run_dir),
         "--text", str(tmp / "cache/text.pt"),
         "--image-prototypes", str(tmp / "cache/img.pt"),
         "--label-prototypes", str(tmp / "cache/lbl.pt"),
         "--targets", str(tmp / "data/targets.csv"), cwd=ROOT)
    _run("src.evaluate", "--run", str(run_dir), *common, cwd=ROOT)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    # predicted_k is chosen, not forced to gold.
    assert metrics["predicted_k"].between(2, 4).all()
