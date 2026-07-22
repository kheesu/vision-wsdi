"""Score clustering assignments and run the go/no-go statistics.

Outputs (in the run directory):
    metrics.csv     long form: one row per method/lambda/seed/lemma with all metrics
    per_lemma.csv   mean/std over seeds per (lemma, method, lambda)
    macro.csv       macro-average over lemmas per (method, lambda, subset scope)
    bootstrap.json  paired bootstrap CIs for the image deltas
    summary.json    go/no-go decision inputs consumed by report.py
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.pilotlib.config import load_config
from src.pilotlib.metrics import all_metrics, paired_bootstrap

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VISUAL_SUBSETS = ("multi_visual", "visual_nonvisual")
_METRIC_COLS = ["ari", "v_measure", "bcubed_p", "bcubed_r", "bcubed_f1"]


def _long_metrics(asg: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for row in asg.to_dict("records"):
        m = all_metrics(row["gold"], row["pred"])
        recs.append(
            {
                "lemma": row["lemma"], "subset": row["subset"], "method": row["method"],
                "lambda": row["lambda"], "seed": row["seed"],
                "n_occurrences": row["n_occurrences"], "gold_k": row["gold_k"],
                "predicted_k": row["predicted_k"],
                "k_correct": int(row["predicted_k"] == row["gold_k"]),
                **m,
            }
        )
    return pd.DataFrame(recs)


def _per_lemma(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby(["lemma", "subset", "method", "lambda"], as_index=False)
    agg = g.agg(
        ari_mean=("ari", "mean"), ari_std=("ari", "std"),
        v_measure_mean=("v_measure", "mean"),
        bcubed_f1_mean=("bcubed_f1", "mean"),
        predicted_k_mean=("predicted_k", "mean"),
        gold_k=("gold_k", "first"), n_occurrences=("n_occurrences", "first"),
    )
    return agg.fillna({"ari_std": 0.0})


def _lolo_tuned_ari(pl: pd.DataFrame, method: str, lemmas: list[str],
                    lambdas: list[float]) -> dict[str, float]:
    """Leave-one-lemma-out lambda selection; returns held-out ari per lemma."""
    tuned = {}
    idx = pl.set_index(["lemma", "method", "lambda"])["ari_mean"]
    for h in lemmas:
        best_lam, best_score = lambdas[0], -np.inf
        for lam in lambdas:
            others = [idx.get((other, method, lam), np.nan) for other in lemmas if other != h]
            score = np.nanmean(others) if others else np.nan
            if np.isfinite(score) and score > best_score:
                best_lam, best_score = lam, score
        tuned[h] = float(idx.get((h, method, best_lam), np.nan))
    return tuned


def _macro(pl: pd.DataFrame) -> pd.DataFrame:
    """Macro-average over lemmas, per (method, lambda) and per subset scope."""
    rows = []
    for scope, sub in [("all", pl),
                       ("visual", pl[pl["subset"].isin(VISUAL_SUBSETS)]),
                       ("multi_visual", pl[pl["subset"] == "multi_visual"]),
                       ("visual_nonvisual", pl[pl["subset"] == "visual_nonvisual"]),
                       ("text_only", pl[pl["subset"] == "text_only"])]:
        if sub.empty:
            continue
        g = sub.groupby(["method", "lambda"], as_index=False).agg(
            macro_ari=("ari_mean", "mean"), macro_v=("v_measure_mean", "mean"),
            macro_bcubed_f1=("bcubed_f1_mean", "mean"),
            k_acc=("predicted_k_mean", "mean"), n_lemmas=("lemma", "nunique"),
        )
        g.insert(0, "scope", scope)
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate clustering runs.")
    ap.add_argument("--run", default="results/oracle_k")
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--output", default=None, help="metrics.csv path (default <run>/metrics.csv)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run = Path(args.run)
    asg = pd.read_parquet(run / "assignments.parquet")

    long = _long_metrics(asg)
    out_metrics = Path(args.output) if args.output else run / "metrics.csv"
    ordered = ["lemma", "subset", "method", "lambda", "seed", "n_occurrences",
               "gold_k", "predicted_k", "ari", "v_measure", "bcubed_f1"]
    extra = [c for c in long.columns if c not in ordered]
    long[ordered + extra].to_csv(out_metrics, index=False)

    pl = _per_lemma(long)
    pl.to_csv(run / "per_lemma.csv", index=False)
    macro = _macro(pl)
    macro.to_csv(run / "macro.csv", index=False)

    # ---- go/no-go: tuned image deltas over visual lemmas -------------------
    lambdas = [float(x) for x in cfg.fusion.lambdas]
    fusion_present = set(pl["method"]) & {"qwen+image", "qwen+label", "qwen+shuffled-image"}
    visual_lemmas = sorted(
        pl[(pl["subset"].isin(VISUAL_SUBSETS)) & (pl["method"] == "qwen+image")]["lemma"].unique()
    ) if "qwen+image" in fusion_present else []

    summary: dict = {"mode": json.loads((run / "run.json").read_text())["mode"],
                     "n_visual_lemmas": len(visual_lemmas)}
    bootstrap: dict = {}

    if visual_lemmas:
        base = pl.set_index(["lemma", "method", "lambda"])["ari_mean"]
        subset_of = {lem: pl[pl["lemma"] == lem]["subset"].iloc[0] for lem in visual_lemmas}
        qwen_ari = {lem: float(base.get((lem, "qwen", -1.0), np.nan)) for lem in visual_lemmas}
        tuned_img = _lolo_tuned_ari(pl, "qwen+image", visual_lemmas, lambdas)
        tuned_lbl = (_lolo_tuned_ari(pl, "qwen+label", visual_lemmas, lambdas)
                     if "qwen+label" in fusion_present
                     else {lem: np.nan for lem in visual_lemmas})
        tuned_shuf = (_lolo_tuned_ari(pl, "qwen+shuffled-image", visual_lemmas, lambdas)
                      if "qwen+shuffled-image" in fusion_present
                      else {lem: np.nan for lem in visual_lemmas})

        # Primary question — is the visual channel meaningful? The anchor profile
        # ALONE vs. the same profile with class->prototype identity permuted.
        img_prof = {lem: float(base.get((lem, "image-profile-only", -1.0), np.nan))
                    for lem in visual_lemmas}
        shuf_prof = {lem: float(base.get((lem, "shuffled-profile-only", -1.0), np.nan))
                     for lem in visual_lemmas}
        d_signal = np.array([img_prof[lem] - shuf_prof[lem] for lem in visual_lemmas])

        d_image = np.array([tuned_img[lem] - qwen_ari[lem] for lem in visual_lemmas])
        d_label = np.array([tuned_img[lem] - tuned_lbl[lem] for lem in visual_lemmas])
        nrs = int(cfg.evaluation.bootstrap_resamples)
        bootstrap = {
            "delta_profile_signal": paired_bootstrap(d_signal, nrs, seed=cfg.seed),
            "delta_image_vs_qwen": paired_bootstrap(d_image, nrs, seed=cfg.seed),
            "delta_image_vs_label": paired_bootstrap(d_label, nrs, seed=cfg.seed),
            "per_lemma": {
                lem: {"qwen": qwen_ari[lem], "qwen+image": tuned_img[lem],
                      "qwen+label": tuned_lbl[lem], "qwen+shuffled-image": tuned_shuf[lem],
                      "image_profile": img_prof[lem], "shuffled_profile": shuf_prof[lem],
                      "delta_image": float(tuned_img[lem] - qwen_ari[lem]),
                      "subset": subset_of[lem]}
                for lem in visual_lemmas
            },
        }

        multi = [lem for lem in visual_lemmas if subset_of[lem] == "multi_visual"]
        pos = np.clip(d_image, 0, None)
        summary.update(
            macro_ari_image_profile=float(np.nanmean(list(img_prof.values()))),
            macro_ari_shuffled_profile=float(np.nanmean(list(shuf_prof.values()))),
            delta_profile_signal=float(np.nanmean(d_signal)),
            frac_visual_profile_beats_null=float(
                np.mean([img_prof[lem] > shuf_prof[lem] for lem in visual_lemmas])),
            macro_ari_qwen=float(np.nanmean(list(qwen_ari.values()))),
            macro_ari_qwen_image=float(np.nanmean(list(tuned_img.values()))),
            macro_ari_qwen_label=float(np.nanmean(list(tuned_lbl.values()))),
            macro_ari_qwen_shuffled=float(np.nanmean(list(tuned_shuf.values()))),
            delta_image=float(np.nanmean(d_image)),
            frac_multi_visual_improved=(
                float(np.mean([tuned_img[lem] - qwen_ari[lem] > 0 for lem in multi]))
                if multi else None),
            max_single_lemma_share=(
                float(np.nanmax(pos) / pos.sum()) if pos.sum() > 0 else None),
        )

    (run / "bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote metrics/per_lemma/macro/bootstrap/summary to %s", run)
    if visual_lemmas:
        logger.info("visual signal: image-profile=%.4f vs shuffled-null=%.4f (Δ=%.4f)",
                    summary["macro_ari_image_profile"], summary["macro_ari_shuffled_profile"],
                    summary["delta_profile_signal"])
        logger.info("naive fusion: qwen=%.4f  qwen+image=%.4f  qwen+label=%.4f",
                    summary["macro_ari_qwen"], summary["macro_ari_qwen_image"],
                    summary["macro_ari_qwen_label"])


if __name__ == "__main__":
    main()
