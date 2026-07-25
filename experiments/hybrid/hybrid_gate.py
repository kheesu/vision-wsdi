"""Hybrid text/image gate analysis (word-level), CPU-only, from saved results.

For each corpus, over the multi_visual lemmas (the assignment-testable set) at 21k:
  text_ari  = mean-over-seeds ARI of `qwen`            (text-only baseline)
  img_ari   = mean-over-seeds ARI of `anchor-assignment` (image grounding)
  uncert    = 1 - mean pairwise cross-seed ARI of the qwen partitions
              (a GOLD-FREE signal for "text is unsure about this word")

Then:
  - oracle ceiling  = mean_lemma max(text, img)       (best a per-word gate could do)
  - always-image    = mean_lemma img
  - best-threshold gate: pick image when uncert>thr (thr swept to maximize realized
    macro ARI). This is the *optimistic* upper bound of the realistic gate family —
    it still uses gold only to pick the single best threshold, nothing per-word.
  - corr(uncert, img-text): if <=0 the uncertainty signal cannot find the rescue words.
"""
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
from scipy.stats import spearmanr

CORP=["semcor","dwug_en","semeval2013","semeval2010"]
SUBSET="multi_visual"

def cross_seed_uncert(a_lem):
    """mean pairwise ARI across the qwen partitions of the 10 seeds -> 1-that."""
    preds=[np.array(p) for p in a_lem.sort_values("seed").pred.tolist()]
    if len(preds)<2: return 0.0
    vals=[ari(preds[i],preds[j]) for i in range(len(preds)) for j in range(i+1,len(preds))]
    return 1.0-float(np.mean(vals))

summary=[]
for corp in CORP:
    pl=pd.read_csv(f"results/{corp}_21k_oracle_k/per_lemma.csv")
    a =pd.read_parquet(f"results/{corp}_21k_oracle_k/assignments.parquet")
    pl=pl[pl.subset==SUBSET]
    lemmas=sorted(pl[pl.method=="qwen"].lemma.unique())
    rows=[]
    for lem in lemmas:
        t=pl[(pl.lemma==lem)&(pl.method=="qwen")].ari_mean
        im=pl[(pl.lemma==lem)&(pl.method=="anchor-assignment")].ari_mean
        if t.empty or im.empty: continue
        t=float(t.iloc[0]); im=float(im.iloc[0])
        u=cross_seed_uncert(a[(a.lemma==lem)&(a.subset==SUBSET)&(a.method=="qwen")])
        rows.append((lem,t,im,u))
    R=pd.DataFrame(rows,columns=["lemma","text","img","uncert"])
    text_only=R.text.mean()
    oracle   =R[["text","img"]].max(1).mean()
    all_img  =R.img.mean()
    # best-threshold gate (optimistic): choose image when uncert>thr
    thrs=sorted(R.uncert.unique())+[1.01]
    best=(-9,None)
    for thr in thrs:
        gate=np.where(R.uncert>thr, R.img, R.text)
        m=gate.mean()
        if m>best[0]: best=(m,thr)
    gate_macro,gate_thr=best
    delta=R.img-R.text
    rho=spearmanr(R.uncert,delta).correlation if len(R)>2 else float("nan")
    rescued=R[delta>0.02].sort_values("uncert",ascending=False)
    summary.append(dict(corpus=corp,n=len(R),text=text_only,oracle=oracle,
        oracle_gain=oracle-text_only,all_image=all_img,
        gate=gate_macro,gate_gain=gate_macro-text_only,gate_thr=gate_thr,
        corr_uncert_delta=rho,n_rescue=int((delta>0.02).sum())))
    print(f"\n=== {corp} (multi_visual, n={len(R)}) ===")
    print(f"  text-only      {text_only:.3f}")
    print(f"  oracle ceiling {oracle:.3f}  (+{oracle-text_only:.3f})")
    print(f"  always-image   {all_img:.3f}  ({all_img-text_only:+.3f})")
    print(f"  best-thr gate  {gate_macro:.3f}  ({gate_macro-text_only:+.3f})  thr={gate_thr:.3f}")
    print(f"  corr(uncert, img-text) = {rho:+.3f}   (<=0 => gate can't find rescue words)")
    if len(rescued):
        print(f"  rescue words (img-text>0.02), by uncert rank among {len(R)}:")
        order=R.sort_values("uncert",ascending=False).reset_index(drop=True)
        for _,r in rescued.iterrows():
            rank=order.index[order.lemma==r.lemma][0]+1
            print(f"    {r.lemma:12s} img {r.img:.2f} text {r.text:.2f}  d{r.img-r.text:+.2f}  uncert {r.uncert:.2f} (rank {rank}/{len(R)})")

print("\n\n==== SUMMARY ====")
S=pd.DataFrame(summary)
print(S.round(3).to_string(index=False))
print(f"\nPooled oracle gain: {(S.oracle_gain*S.n).sum()/S.n.sum():+.4f} | "
      f"pooled best-gate gain: {(S.gate_gain*S.n).sum()/S.n.sum():+.4f}")
