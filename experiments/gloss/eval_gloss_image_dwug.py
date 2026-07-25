"""Gloss->image retrieval anchors on DWUG. For each sense gloss, retrieve the
top-k nearest ImageNet class prototypes (cross-modal, one Qwen space) and average
them into a visual anchor. Compare assignment by gloss-TEXT vs gloss-IMAGE vs a
blend, over the 21 DWUG words. Also prints what each gloss retrieved.
"""
import numpy as np, pandas as pd, torch
from sklearn.metrics import adjusted_rand_score
from src.pilotlib.config import load_config
from src.pilotlib.embedders import QwenEmbedder
from src.pilotlib.wordnet_utils import wnid_to_synset

OCC="data/dwug_occurrences.parquet"; GLOSS="data/dwug_glosses.csv"; BANK="cache/img_bank.pt"
MINOCC,MINSENSE,MINPER=40,2,8; L=R=20; K=5; ALPHAS=[0.0,0.25,0.5,0.75,1.0]

def window(s,a,b):
    w,sp,c=[],[],0
    for tok in s.split(" "):
        w.append(tok); sp.append((c,c+len(tok))); c+=len(tok)+1
    tg=[i for i,(x,y) in enumerate(sp) if x<b and y>a]
    return s if not tg else " ".join(w[max(0,tg[0]-L):min(len(w),tg[-1]+1+R)])

occ=pd.read_parquet(OCC); gl=pd.read_csv(GLOSS)
glmap={(r.lemma,int(r.cluster)):r.gloss for r in gl.itertuples()}
bk=torch.load(BANK,weights_only=False)["prototypes"]
bw=list(bk); B=np.stack([bk[w].numpy() for w in bw])                     # (Nbank,4096) unit
cfg=load_config("configs/pilot.yaml")
emb=QwenEmbedder(cfg.models.embedding, dtype=cfg.embedding.get("dtype","bfloat16"),
                 prompt=cfg.embedding.get("prompt",None))

lemmas=[]
for lem,g in occ.groupby("lemma"):
    vc=g.gold_synset.value_counts(); ret=vc[vc>=MINPER]
    if len(ret)<MINSENSE or int(ret.sum())<MINOCC: continue
    cls=[int(s.split(".cl")[1]) for s in ret.index]
    if all((lem,c) in glmap for c in cls): lemmas.append((lem,ret.index.tolist(),cls))
print(f"words: {len(lemmas)} | bank: {len(bw)} classes | k={K}\n")

def retrieve(gvec):
    idx=np.argsort(-(B@gvec))[:K]
    v=B[idx].mean(0); return v/(np.linalg.norm(v)+1e-12), [bw[i] for i in idx]

rows=[]; show={"plane","ball","land","grain"}
for lem,senses,cls in lemmas:
    sub=occ[(occ.lemma==lem)&(occ.gold_synset.isin(senses))].reset_index(drop=True)
    t=emb.encode_texts([window(r.sentence,int(r.target_start),int(r.target_end)) for r in sub.itertuples()],
                       batch_size=int(cfg.embedding.text_batch_size))
    G=emb.encode_texts([glmap[(lem,c)] for c in cls], batch_size=int(cfg.embedding.text_batch_size))
    V=[];
    for ci,c in enumerate(cls):
        v,got=retrieve(G[ci]); V.append(v)
        if lem in show:
            names=[ (wnid_to_synset(w).name() if wnid_to_synset(w) else w) for w in got[:3]]
            print(f"  {lem}.cl{c}: “{glmap[(lem,c)][:52]}” -> {names}")
    V=np.stack(V)
    gmap={c:i for i,c in enumerate(cls)}
    gold=np.array([gmap[int(s.split('.cl')[1])] for s in sub.gold_synset])
    St=t@G.T; Si=t@V.T
    ari_t=adjusted_rand_score(gold,St.argmax(1))
    ari_i=adjusted_rand_score(gold,Si.argmax(1))
    best=(-9,None)
    for a in ALPHAS:
        ari=adjusted_rand_score(gold,(a*St+(1-a)*Si).argmax(1))
        if ari>best[0]: best=(ari,a)
    rows.append((lem,len(cls),ari_t,ari_i,best[0],best[1]))

R=pd.DataFrame(rows,columns=["lemma","S","gloss_text","gloss_image","blend_best","alpha*"]).sort_values("gloss_text",ascending=False)
print("\n"+R.round(3).to_string(index=False))
print(f"\nMACRO  gloss-text={R.gloss_text.mean():.3f}  gloss-image={R.gloss_image.mean():.3f}  blend(best-α per word)={R.blend_best.mean():.3f}")
print(f"words where image>text: {int((R.gloss_image>R.gloss_text).sum())}/{len(R)}")
