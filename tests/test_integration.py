"""End-to-end test of cluster -> evaluate -> report on synthetic caches.

Builds fixtures where the visual anchor profile carries a strong sense signal,
BERT is only weakly separable, the label anchor is intermediate, and the
shuffled control is uninformative. The pipeline should then produce a GO with
ARI(bert+image) > ARI(bert+label) > ARI(bert) ~ ARI(shuffled).
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
D_CLIP = 16
H_BERT = 32


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _build_fixtures(tmp: Path, seed: int = 0):
    rng = np.random.RandomState(seed)
    lemmas = ["alpha", "beta", "gamma", "delta"]
    per_sense = 24

    # Two orthonormal-ish visual concepts per lemma live in the CLIP space.
    bert_rows, clip_rows, meta = [], [], []
    img_proto, lbl_proto = {}, {}
    targ_rows = []

    for li, lemma in enumerate(lemmas):
        w0, w1 = f"n{li:08d}", f"n{li+100:08d}"
        v0 = _unit(rng.normal(size=D_CLIP))
        v1 = _unit(rng.normal(size=D_CLIP))
        img_proto[w0], img_proto[w1] = v0, v1
        # Label prototypes: weaker sense signal (rotate toward the shared mean).
        mean = _unit(v0 + v1)
        lbl_proto[w0] = _unit(0.5 * v0 + 0.5 * mean)
        lbl_proto[w1] = _unit(0.5 * v1 + 0.5 * mean)

        # BERT: two heavily overlapping blobs (weak text signal).
        c0 = rng.normal(size=H_BERT)
        c1 = c0 + 0.35 * rng.normal(size=H_BERT)
        for sense_idx, (v, c) in enumerate([(v0, c0), (v1, c1)]):
            syn = f"{lemma}.n.0{sense_idx + 1}"
            for _ in range(per_sense):
                clip_rows.append(_unit(v + 0.55 * rng.normal(size=D_CLIP)))
                bert_rows.append(c + 1.4 * rng.normal(size=H_BERT))
                meta.append({"lemma": lemma, "sentence_id": len(meta),
                             "gold_synset": syn, "subset": "multi_visual"})
        targ_rows.append({
            "lemma": lemma, "subset": "multi_visual", "n_occurrences": 2 * per_sense,
            "n_senses": 2, "gold_k": 2, "n_visual_anchors": 2,
            "anchor_wnids": f"{w0};{w1}",
            "retained_senses": f"{lemma}.n.01;{lemma}.n.02",
        })

    bert = np.asarray(bert_rows, dtype=np.float32)
    clip = np.asarray(clip_rows, dtype=np.float32)

    (tmp / "cache").mkdir(parents=True, exist_ok=True)
    (tmp / "data").mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": bert, "meta": meta, "model": "fake-bert"}, tmp / "cache/bert.pt")
    torch.save({"vectors": clip, "vectors_raw": None, "meta": meta, "model": "fake-clip"},
               tmp / "cache/clip.pt")
    torch.save({"prototypes": {k: torch.from_numpy(v.astype(np.float32))
                               for k, v in img_proto.items()}, "dim": D_CLIP},
               tmp / "cache/img.pt")
    torch.save({"prototypes": {k: torch.from_numpy(v.astype(np.float32))
                               for k, v in lbl_proto.items()}}, tmp / "cache/lbl.pt")
    pd.DataFrame(targ_rows).to_csv(tmp / "data/targets.csv", index=False)

    cfg = {
        "seed": 13,
        "fusion": {"lambdas": [0.5, 1.0, 2.0]},
        "contexts": {"pca_dimensions": 64},
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
         "--bert", str(tmp / "cache/bert.pt"), "--clip", str(tmp / "cache/clip.pt"),
         "--image-prototypes", str(tmp / "cache/img.pt"),
         "--label-prototypes", str(tmp / "cache/lbl.pt"),
         "--targets", str(tmp / "data/targets.csv"), cwd=ROOT)
    _run("src.evaluate", "--run", str(run_dir), *common, cwd=ROOT)
    _run("src.report", "--run", str(run_dir), cwd=ROOT)

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["n_visual_lemmas"] == 4
    assert summary["macro_ari_bert_image"] > summary["macro_ari_bert"]
    assert summary["macro_ari_bert_image"] > summary["macro_ari_bert_label"]
    # Shuffled control must not match the real image gain.
    assert summary["macro_ari_bert_image"] > summary["macro_ari_bert_shuffled"]

    bootstrap = json.loads((run_dir / "bootstrap.json").read_text())
    assert bootstrap["delta_image_vs_bert"]["point"] > 0
    assert (run_dir / "report.md").exists()
    assert (run_dir / "metrics.csv").exists()


def test_unknown_k_runs(tmp_path):
    tmp = _build_fixtures(tmp_path)
    run_dir = tmp / "results/unknown_k"
    common = ["--config", str(tmp / "cfg.yaml")]
    _run("src.cluster", *common, "--mode", "unknown", "--output", str(run_dir),
         "--bert", str(tmp / "cache/bert.pt"), "--clip", str(tmp / "cache/clip.pt"),
         "--image-prototypes", str(tmp / "cache/img.pt"),
         "--label-prototypes", str(tmp / "cache/lbl.pt"),
         "--targets", str(tmp / "data/targets.csv"), cwd=ROOT)
    _run("src.evaluate", "--run", str(run_dir), *common, cwd=ROOT)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    # predicted_k is chosen, not forced to gold.
    assert metrics["predicted_k"].between(2, 4).all()
