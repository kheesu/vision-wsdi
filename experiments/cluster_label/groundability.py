"""Intrinsic, gold-free groundability predictor.

For each multi_visual word, build one prototype per grounded sense (normalized mean
of that sense's ImageNet anchor-class prototypes, from the 2115-class img_bank), then
measure how distinguishable the senses are AS IMAGES:
    collapse   = mean pairwise cosine between sense prototypes  (high => senses look alike)
    worst_pair = max  pairwise cosine                           (the most confusable pair)
    separability = 1 - collapse
This uses NO gold labels and NO text. Validate against the §6.7 outcome
Δ = anchor-assignment ARI - null: does low collapse predict the words that work?
"""
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr

BANK = torch.load("cache/img_bank.pt", weights_only=False)["prototypes"]
bank = {w: (v.numpy() if hasattr(v, "numpy") else np.asarray(v)) for w, v in BANK.items()}
def unit(v): return v / (np.linalg.norm(v) + 1e-12)

TGT = "/home/heesu/.claude/jobs/dcce83f7/tmp/targets"
CORP = ["semcor", "dwug_en", "semeval2013", "semeval2010"]

def grouping(row):
    g = {}
    for part in str(row.anchor_grouping).split("|"):
        if "=" in part:
            s, ws = part.split("=", 1)
            cols = [w for w in ws.split(",") if w]
            if cols: g[s] = cols
    return g

rows = []
for corp in CORP:
    tg = pd.read_csv(f"{TGT}/{corp}.csv")
    tg = tg[tg.subset == "multi_visual"]
    pl = pd.read_csv(f"results/{corp}_21k_oracle_k/per_lemma.csv"); pl = pl[pl.subset == "multi_visual"]
    for r in tg.itertuples(index=False):
        g = grouping(r)
        protos, miss = {}, 0
        for s, ws in g.items():
            vs = [bank[w] for w in ws if w in bank]
            miss += sum(w not in bank for w in ws)
            if vs: protos[s] = unit(np.mean([unit(v) for v in vs], axis=0))
        if len(protos) < 2:
            continue
        P = np.stack(list(protos.values()))
        C = P @ P.T
        iu = np.triu_indices(len(P), k=1)
        collapse = float(C[iu].mean()); worst = float(C[iu].max())
        a = pl[(pl.lemma == r.lemma) & (pl.method == "anchor-assignment")].ari_mean
        n = pl[(pl.lemma == r.lemma) & (pl.method == "anchor-assignment-shuffled")].ari_mean
        if a.empty or n.empty: continue
        rows.append((corp, r.lemma, len(protos), collapse, worst,
                     float(a.iloc[0]) - float(n.iloc[0])))

R = pd.DataFrame(rows, columns=["corp", "lemma", "n_sense", "collapse", "worst_pair", "delta"])
R["sep"] = 1 - R.collapse
R["works"] = R.delta >= 0.10
print(f"words scored: {len(R)}  | works: {int(R.works.sum())}")
print(f"\nmean pairwise anchor cosine (collapse):  works={R[R.works].collapse.mean():.3f}   fails={R[~R.works].collapse.mean():.3f}")
print(f"worst-pair cosine:                        works={R[R.works].worst_pair.mean():.3f}   fails={R[~R.works].worst_pair.mean():.3f}")
print(f"\nSpearman(collapse,   delta) = {spearmanr(R.collapse, R.delta).correlation:+.3f}")
print(f"Spearman(worst_pair, delta) = {spearmanr(R.worst_pair, R.delta).correlation:+.3f}")

print("\n== gold-free selection rule: keep words with collapse < thr ==")
for thr in [0.55, 0.60, 0.65, 0.70]:
    sel = R.collapse < thr
    if sel.sum() == 0: continue
    tp = int((sel & R.works).sum()); fp = int((sel & ~R.works).sum()); fn = int((~sel & R.works).sum())
    prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
    kept_delta = R[sel].delta.mean(); rej_delta = R[~sel].delta.mean()
    print(f"  collapse<{thr:.2f}: keep {sel.sum():2d}/{len(R)}  prec {prec:.2f} rec {rec:.2f}  |  "
          f"mean Δ kept {kept_delta:+.3f}  vs rejected {rej_delta:+.3f}")

print("\n== most collapsed (predicted NOT groundable) ==")
for _, r in R.sort_values("collapse", ascending=False).head(8).iterrows():
    print(f"  {r.lemma:11s}({r.corp[:3]}) collapse {r.collapse:.2f} worst {r.worst_pair:.2f}  Δ {r.delta:+.2f}  {'WORKS' if r.works else 'fails'}")
print("== least collapsed (predicted groundable) ==")
for _, r in R.sort_values("collapse").head(8).iterrows():
    print(f"  {r.lemma:11s}({r.corp[:3]}) collapse {r.collapse:.2f} worst {r.worst_pair:.2f}  Δ {r.delta:+.2f}  {'WORKS' if r.works else 'fails'}")
R.to_csv("experiments/cluster_label/groundability.csv", index=False)
print("\nwrote experiments/cluster_label/groundability.csv")
