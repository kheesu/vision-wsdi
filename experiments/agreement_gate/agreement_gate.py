"""Text<->image agreement as a gold-free groundability gate, plus two audits of §12.

Reads the saved 21k oracle-K run artifacts (assignments.parquet + per_lemma.csv)
for all four corpora — CPU-only, no cache/embedding needed. Run from a checkout
whose results/ holds the `<corpus>_21k_oracle_k` runs (results/* is git-ignored,
so that is the checkout the pipeline ran in), or pass --results-root.

Per multi_visual word it computes, gold-free:
    max_share  = share of usages the anchor assignment routes to its top sense
                 (§12's behavioral signal, recomputed from the saved partitions)
    agreement  = mean over qwen seeds of ARI(text partition, anchor assignment)
                 (the new signal: use the strong text partition as pseudo-gold)
and validates both against the gold outcome Δ = assignment ARI − shuffled null.

Three questions:
  1. Is §12's max-share correlation partly mechanical? (a collapsed assignment
     is a near-one-cluster partition, so its ARI ~ 0 by construction)
  2. Does agreement predict Δ better, and what does a combined gate buy?
  3. Quadrants: where do the "rescue" words (image beats text) live — can any
     gold-free signal find them?

Outputs agreement_gate.csv next to this script.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score as ari

HERE = Path(__file__).resolve().parent
CORPORA = ["semcor", "dwug_en", "semeval2013", "semeval2010"]
WORKS = 0.10        # §6.7 "clearly works" bar on Δ
COLLAPSE = 0.80     # §12 behavioral gate threshold on max_share
AGREE = 0.10        # agreement gate threshold on ARI(text, assignment)

ap = argparse.ArgumentParser()
ap.add_argument("--results-root", default="results", help="dir holding <corpus>_21k_oracle_k runs")
args = ap.parse_args()
root = Path(args.results_root)

rows = []
for corp in CORPORA:
    run = root / f"{corp}_21k_oracle_k"
    A = pd.read_parquet(run / "assignments.parquet")
    PL = pd.read_csv(run / "per_lemma.csv")
    for lem, g in A[A.subset == "multi_visual"].groupby("lemma"):
        aa = g[g.method == "anchor-assignment"]
        if aa.empty:
            continue
        pred = np.asarray(aa.iloc[0]["pred"])           # deterministic across seeds
        gold = np.asarray(aa.iloc[0]["gold"])
        _, cnt = np.unique(pred, return_counts=True)
        qn = g[g.method == "qwen"]
        agree = float(np.mean([ari(np.asarray(r["pred"]), pred) for _, r in qn.iterrows()]))
        pl = PL[(PL.lemma == lem) & (PL.subset == "multi_visual")]

        def m(meth):
            s = pl[pl.method == meth].ari_mean
            return float(s.iloc[0]) if len(s) else np.nan

        a, null, text = m("anchor-assignment"), m("anchor-assignment-shuffled"), m("qwen")
        _, gcnt = np.unique(gold, return_counts=True)
        rows.append(dict(corp=corp, lemma=lem, n=len(pred),
                         max_share=cnt.max() / cnt.sum(), n_pred_used=len(cnt),
                         agree=agree, assign_ari=a, null_ari=null, text_ari=text,
                         delta=a - null, img_minus_text=a - text,
                         gold_max_share=gcnt.max() / gcnt.sum()))

R = pd.DataFrame(rows)
works = R.delta >= WORKS
spread = R.max_share < COLLAPSE
print(f"multi_visual words: {len(R)}   works (Δ>={WORKS}): {int(works.sum())}")

print("\n== 1. §12 behavioral signal: reproduction + circularity audit ==")
print(f"Spearman(max_share, Δ)        = {spearmanr(R.max_share, R.delta).correlation:+.3f}"
      "   (report: -0.40)")
print(f"Spearman(max_share, assign)   = {spearmanr(R.max_share, R.assign_ari).correlation:+.3f}")
print(f"collapsed (share>={COLLAPSE}): n={int((~spread).sum())}  "
      f"assign ARI>0.05 on only {int((R.loc[~spread, 'assign_ari'] > 0.05).sum())} of them "
      f"(near-degenerate partition => ARI~0 by construction)")
print(f"within spread words only:     Spearman(max_share, Δ) = "
      f"{spearmanr(R.loc[spread, 'max_share'], R.loc[spread, 'delta']).correlation:+.3f}  "
      f"works-rate {float(works[spread].mean()):.2f} "
      f"(vs {float(works[~spread].mean()):.2f} collapsed)")

print("\n== 2. gold-free agreement signal: ARI(text partition, anchor assignment) ==")
print(f"Spearman(agree, Δ)            = {spearmanr(R.agree, R.delta).correlation:+.3f}")
print(f"Spearman(agree, assign)       = {spearmanr(R.agree, R.assign_ari).correlation:+.3f}")
print(f"Spearman(agree, img−text)     = {spearmanr(R.agree, R.img_minus_text).correlation:+.3f}"
      "   (rescues stay invisible)")


def gate_row(name, sel):
    tp = int((sel & works).sum())
    prec = tp / int(sel.sum()) if sel.sum() else 0.0
    rec = tp / int(works.sum())
    print(f"  {name:34s} keep {int(sel.sum()):2d}/{len(R)}  prec {prec:.2f}  rec {rec:.2f}  "
          f"mean Δ kept {R.loc[sel, 'delta'].mean():+.3f}  "
          f"rejected {R.loc[~sel, 'delta'].mean():+.3f}")


print("\n== gate comparison ==")
gate_row(f"§12 behavioral (share<{COLLAPSE})", spread)
gate_row(f"combined (share<{COLLAPSE} & agree>{AGREE})", spread & (R.agree > AGREE))

print("\n== 3. quadrants (rescue = img_minus_text > 0) ==")
R["quad"] = "collapsed"
R.loc[spread & (R.agree > AGREE), "quad"] = "spread+agree"
R.loc[spread & (R.agree <= AGREE), "quad"] = "spread+disagree"
q = R.groupby("quad").agg(n=("delta", "size"), mean_delta=("delta", "mean"),
                          works=("delta", lambda s: int((s >= WORKS).sum())),
                          rescues=("img_minus_text", lambda s: int((s > 0).sum())))
print(q.round(3).to_string())
resc = R.img_minus_text > 0
print(f"rescue words captured by combined gate: "
      f"{int((resc & (R.quad == 'spread+agree')).sum())} of {int(resc.sum())}")

print("\n== inflated shuffled nulls (null ARI > 0.15) ==")
print(R[R.null_ari > 0.15][["corp", "lemma", "n", "assign_ari", "null_ari",
                            "gold_max_share"]].round(3).to_string(index=False))

out = HERE / "agreement_gate.csv"
R.round(6).to_csv(out, index=False)
print(f"\nwrote {out}")
