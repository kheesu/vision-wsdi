"""Full per-word assignment inventory, conservative thresholds, from saved results.

multi_visual lemmas at 21k (oracle-K). assign = anchor-assignment ARI (mean/seeds),
null = anchor-assignment-shuffled ARI. delta = assign - null.
  works    delta >= 0.10
  marginal 0.02 <= delta < 0.10
  no       delta < 0.02   (conservative: tiny positive delta = no reliable signal)
"""
import pandas as pd
CORP=["semcor","dwug_en","semeval2013","semeval2010"]
rows=[]
for corp in CORP:
    pl=pd.read_csv(f"results/{corp}_21k_oracle_k/per_lemma.csv")
    pl=pl[pl.subset=="multi_visual"]
    for lem in sorted(pl[pl.method=="anchor-assignment"].lemma.unique()):
        a=pl[(pl.lemma==lem)&(pl.method=="anchor-assignment")].ari_mean
        n=pl[(pl.lemma==lem)&(pl.method=="anchor-assignment-shuffled")].ari_mean
        if a.empty or n.empty: continue
        a=float(a.iloc[0]); n=float(n.iloc[0]); d=a-n
        cls="works" if d>=0.10 else ("marginal" if d>=0.02 else "no")
        rows.append((corp,lem,a,n,d,cls))
R=pd.DataFrame(rows,columns=["corpus","lemma","assign","null","delta","cls"])

print("=== overall counts (conservative) ===")
print(R.cls.value_counts().reindex(["works","marginal","no"]).to_string())
print(f"total pairs: {len(R)}")

print("\n=== per-corpus counts ===")
piv=R.pivot_table(index="corpus",columns="cls",values="lemma",aggfunc="count",fill_value=0)
piv=piv.reindex(columns=["works","marginal","no"],fill_value=0)
piv["total"]=piv.sum(1)
print(piv.reindex(CORP).to_string())

print("\n=== WORKS (delta>=0.10), sorted by delta desc ===")
w=R[R.cls=="works"].sort_values("delta",ascending=False)
for _,r in w.iterrows():
    print(f"  {r.lemma:12s} ({r.corpus:11s}) assign {r.assign:.2f} null {r.null:.2f}  d {r.delta:+.2f}")

print("\n=== NO / fails, most-negative first (worst 12) ===")
f=R[R.cls=="no"].sort_values("delta")
for _,r in f.head(12).iterrows():
    print(f"  {r.lemma:12s} ({r.corpus:11s}) assign {r.assign:.2f} null {r.null:.2f}  d {r.delta:+.2f}")

print("\n=== MARGINAL (0.02<=d<0.10) ===")
m=R[R.cls=="marginal"].sort_values("delta",ascending=False)
print("  " + ", ".join(f"{r.lemma}({r.corpus[:3]},{r.delta:+.2f})" for _,r in m.iterrows()))
