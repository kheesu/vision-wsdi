"""Prototype: gloss-based sense anchors on DWUG-EN (fixed null).

Anchor each gold cluster by its gloss (Qwen text embedding); assign each usage
to argmax cos(context, gloss); score vs gold by ARI + accuracy. NULL = replace
the word's glosses with the SAME NUMBER of random glosses drawn from OTHER words
(real definitions of unrelated senses), averaged over several draws — a valid
control since ARI is invariant to merely relabelling a word's own anchors.
"""
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score
from src.pilotlib.config import load_config
from src.pilotlib.embedders import QwenEmbedder

OCC="data/dwug_occurrences.parquet"; GLOSS="data/dwug_glosses.csv"
MINOCC,MINSENSE,MINPER=40,2,8; L=R=20; NULL_DRAWS=5

def window(s,a,b):
    words,spans,cur=[],[],0
    for tok in s.split(" "):
        words.append(tok); spans.append((cur,cur+len(tok))); cur+=len(tok)+1
    tgt=[i for i,(x,y) in enumerate(spans) if x<b and y>a]
    return s if not tgt else " ".join(words[max(0,tgt[0]-L):min(len(words),tgt[-1]+1+R)])

occ=pd.read_parquet(OCC); gl=pd.read_csv(GLOSS)
glmap={(r.lemma,int(r.cluster)):r.gloss for r in gl.itertuples()}
cfg=load_config("configs/pilot.yaml")
emb=QwenEmbedder(cfg.models.embedding, dtype=cfg.embedding.get("dtype","bfloat16"),
                 prompt=cfg.embedding.get("prompt",None))
# embed the whole gloss pool once
pool_keys=list(glmap); pool_txt=[glmap[k] for k in pool_keys]
Gpool=emb.encode_texts(pool_txt, batch_size=int(cfg.embedding.text_batch_size))
pool_idx={k:i for i,k in enumerate(pool_keys)}

lemmas=[]
for lem,g in occ.groupby("lemma"):
    vc=g.gold_synset.value_counts(); ret=vc[vc>=MINPER]
    if len(ret)<MINSENSE or int(ret.sum())<MINOCC: continue
    cls=[int(s.split(".cl")[1]) for s in ret.index]
    if all((lem,c) in glmap for c in cls): lemmas.append((lem,ret.index.tolist(),cls))
print(f"DWUG words usable with glosses: {len(lemmas)}")

rng=np.random.RandomState(13); rows=[]
for lem,senses,cls in lemmas:
    sub=occ[(occ.lemma==lem)&(occ.gold_synset.isin(senses))].reset_index(drop=True)
    t=emb.encode_texts([window(r.sentence,int(r.target_start),int(r.target_end)) for r in sub.itertuples()],
                       batch_size=int(cfg.embedding.text_batch_size))
    own=[pool_idx[(lem,c)] for c in cls]; G=Gpool[own]
    gmap={c:i for i,c in enumerate(cls)}
    gold=np.array([gmap[int(s.split(".cl")[1])] for s in sub.gold_synset])
    pred=(t@G.T).argmax(1)
    ari=adjusted_rand_score(gold,pred); acc=float((pred==gold).mean())
    # null: random glosses from other words
    other=[i for k,i in pool_idx.items() if k[0]!=lem]
    nulls=[]
    for _ in range(NULL_DRAWS):
        pick=rng.choice(other,size=len(cls),replace=False)
        nulls.append(adjusted_rand_score(gold,(t@Gpool[pick].T).argmax(1)))
    rows.append((lem,len(cls),len(sub),ari,float(np.mean(nulls)),acc))

R=pd.DataFrame(rows,columns=["lemma","S","n","gloss_ARI","null_ARI","accuracy"]).sort_values("gloss_ARI",ascending=False)
print(R.round(3).to_string(index=False))
d=R.gloss_ARI-R.null_ARI
print(f"\nMACRO gloss ARI = {R.gloss_ARI.mean():.3f} | random-gloss null = {R.null_ARI.mean():.3f} | Δ = {d.mean():+.3f}")
print(f"words beating null: {int((d>0).sum())}/{len(R)}")
print(f"MACRO accuracy = {R.accuracy.mean():.3f} (chance ≈ {(1/R.S).mean():.3f})")
print(f"[ref] image-anchor assignment DWUG multi_visual @21k: 0.199 vs 0.106 null (6 words)")
