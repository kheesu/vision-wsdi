"""Cluster-then-label: use TEXT for induction, IMAGE anchors only to NAME clusters.

Runs on whatever corpus is currently cached (cache/text_contexts.pt + data/targets.csv).

Variant (a) — cluster-labeling accuracy (does naming work?):
  cluster usages by text (spherical k-means on PCA-64, oracle K=gold_k), then name
  each cluster by the MAJORITY anchor-sense of its usages (aggregated, not per-usage).
  Accuracy = fraction of usages whose cluster-name == their gold sense name.
  Compared against Q1's per-usage anchor-assignment accuracy (the noisy baseline).
  Reported over all usages and over the "nameable" subset (gold sense is grounded).

Variant (b) — label-and-merge ARI (does naming change the partition?):
  merge text clusters that receive the SAME anchor-name, recompute ARI vs plain text.
  Run on oracle-K (in-process) and unknown-K (saved partition), where over-splitting
  can actually be fixed.

Everything is label-aware via names: gold name = meta.gold_synset; anchor name =
senses_sorted[pred], the sorted grounded-sense keys used by construct_features.
"""
import sys, numpy as np, pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari, silhouette_score
from src.construct_features import FeatureBank
from src.pilotlib.config import load_config

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "semcor"
cfg = load_config("configs/pilot.yaml")
seeds = list(cfg.clustering.seeds)
n_init = int(cfg.clustering.n_init)
K_MIN, K_MAX, K_DEN = int(cfg.clustering.unknown_k_min), int(cfg.clustering.unknown_k_max), int(cfg.clustering.unknown_k_denom)
fb = FeatureBank("cache/text_contexts.pt", "cache/imagenet_prototypes.pt",
                 "cache/label_prototypes.pt", "data/targets.csv",
                 pca_dim=int(cfg.contexts.pca_dimensions))

def l2(x): return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
def skmeans(x, k, seed): return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(l2(x))
def choose_k(x, seed, gold_k):                           # unknown-K: cosine-silhouette search (mirrors cluster.py)
    xn = l2(x); n = len(x); k_hi = min(K_MAX, n // K_DEN)
    if k_hi < K_MIN: return skmeans(x, min(gold_k, n), seed)
    best = (-np.inf, None)
    for k in range(K_MIN, k_hi + 1):
        lab = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(xn)
        if len(np.unique(lab)) < 2: continue
        s = silhouette_score(xn, lab, metric="cosine")
        if s > best[0]: best = (s, lab)
    return best[1] if best[1] is not None else np.zeros(n, dtype=int)

def anchor_assign(lemma, t):
    """Reproduce construct_features anchor-assignment from image protos only
    (bypass the stale label_proto). Returns (anchor_name_per_usage, grounded_senses)."""
    img_anchors = [w for w in fb.anchors.get(lemma, []) if w in fb.image_proto]
    if not img_anchors:
        return None, set()
    a_img = np.stack([t @ fb.image_proto[w] for w in img_anchors], axis=1)   # (n, |anchors|)
    col = {w: i for i, w in enumerate(img_anchors)}
    sc = {}
    for s, ws in fb.grouping.get(lemma, {}).items():
        idxs = [col[w] for w in ws if w in col]
        if idxs: sc[s] = idxs
    if not sc: sc = {w: [i] for i, w in enumerate(img_anchors)}
    ss = sorted(sc)
    scores = np.stack([a_img[:, sc[s]].max(axis=1) for s in ss], axis=1)
    pred = scores.argmax(axis=1)
    return np.array([ss[i] for i in pred]), set(ss)

rows = []
for lemma in fb.lemmas():
    if fb.subset_of.get(lemma) != "multi_visual":
        continue
    ridx = fb.rows[lemma]
    gold_name = fb.meta.loc[ridx, "gold_synset"].to_numpy()          # per-usage gold name
    feats = fb.build(lemma, [], seeds[0])
    gold_ids = feats["gold"]
    t = fb.text[ridx]                                                # raw (unit) Qwen embedding
    anchor_name, grounded = anchor_assign(lemma, t)                  # per-usage anchor name
    if anchor_name is None:
        continue
    nameable = np.array([g in grounded for g in gold_name])
    gold_k = feats["gold_k"]
    h = feats["systems"]["qwen"]
    n = feats["n"]

    # Q1 per-usage labeling accuracy (name matches gold, over nameable usages + all)
    q1_all = float((anchor_name == gold_name).mean())
    q1_nam = float((anchor_name[nameable] == gold_name[nameable]).mean()) if nameable.any() else np.nan

    # Variant (a): cluster then majority-vote name, averaged over seeds
    ctl_all, ctl_nam, merge_o, plain_o = [], [], [], []
    for seed in seeds:
        cl = skmeans(h, min(gold_k, n), seed)
        # name each cluster by majority anchor-sense of its usages
        cname = {}
        for c in np.unique(cl):
            m = cl == c
            cname[c] = Counter(anchor_name[m]).most_common(1)[0][0]
        assigned = np.array([cname[c] for c in cl])                  # per-usage cluster-name
        ctl_all.append(float((assigned == gold_name).mean()))
        if nameable.any():
            ctl_nam.append(float((assigned[nameable] == gold_name[nameable]).mean()))
        # Variant (b) oracle: merge clusters sharing a name -> partition by assigned name
        merge_o.append(ari(gold_ids, pd.factorize(assigned)[0]))
        plain_o.append(ari(gold_ids, cl))

    rec = dict(lemma=lemma, n=n, gold_k=int(gold_k), grounded=len(grounded),
               q1_all=q1_all, q1_nam=q1_nam,
               ctl_all=float(np.mean(ctl_all)),
               ctl_nam=float(np.mean(ctl_nam)) if ctl_nam else np.nan,
               ari_text_o=float(np.mean(plain_o)), ari_merge_o=float(np.mean(merge_o)))

    # Variant (b) unknown-K: cluster in-process (silhouette K search), merge by name
    m_u, p_u = [], []
    for seed in seeds:
        cl = choose_k(h, seed, gold_k)
        cname = {c: Counter(anchor_name[cl == c]).most_common(1)[0][0] for c in np.unique(cl)}
        assigned = np.array([cname[c] for c in cl])
        m_u.append(ari(gold_ids, pd.factorize(assigned)[0]))
        p_u.append(ari(gold_ids, cl))
    rec["ari_text_u"] = float(np.mean(p_u))
    rec["ari_merge_u"] = float(np.mean(m_u))
    rows.append(rec)

R = pd.DataFrame(rows)
print(f"\n################  {CORPUS}  —  {len(R)} multi_visual words  ################")

print("\n===== VARIANT (a): can images correctly NAME text clusters? =====")
print("  (nameable = gold sense is grounded; only defined when gold uses the same")
print("   sense inventory as the anchors, i.e. WordNet-gold corpora like SemCor)")
print(f"  Q1 per-usage anchor labeling accuracy   : all {np.nanmean(R.q1_all):.3f} | nameable {np.nanmean(R.q1_nam):.3f}")
print(f"  cluster-then-label (majority vote)       : all {np.nanmean(R.ctl_all):.3f} | nameable {np.nanmean(R.ctl_nam):.3f}")
print(f"  -> denoising gain (nameable)             : {np.nanmean(R.ctl_nam)-np.nanmean(R.q1_nam):+.3f}")
print("\n  biggest naming improvements (nameable, ctl - q1):")
R["gain_nam"] = R.ctl_nam - R.q1_nam
for _, r in R.sort_values("gain_nam", ascending=False).head(8).iterrows():
    print(f"    {r.lemma:11s} q1 {r.q1_nam:.2f} -> ctl {r.ctl_nam:.2f}  ({r.gain_nam:+.2f})  gold_k={r.gold_k} grounded={r.grounded}")

print("\n===== VARIANT (b): does label-and-merge change induction ARI? =====")
print(f"  oracle-K : text {R.ari_text_o.mean():.3f}  ->  merged {R.ari_merge_o.mean():.3f}  ({R.ari_merge_o.mean()-R.ari_text_o.mean():+.3f})")
U = R.dropna(subset=["ari_text_u"])
print(f"  unknownK : text {U.ari_text_u.mean():.3f}  ->  merged {U.ari_merge_u.mean():.3f}  ({U.ari_merge_u.mean()-U.ari_text_u.mean():+.3f})  (n={len(U)})")
U2 = U.copy(); U2["d"] = U2.ari_merge_u - U2.ari_text_u
helped = U2[U2.d > 0.02]; hurt = U2[U2.d < -0.02]
print(f"    merge helped {len(helped)} words (e.g. {', '.join(helped.sort_values('d',ascending=False).lemma.head(5))})")
print(f"    merge hurt   {len(hurt)} words (e.g. {', '.join(hurt.sort_values('d').lemma.head(5))})")

R.to_csv(f"experiments/cluster_label/results_{CORPUS}.csv", index=False)
print(f"\nwrote experiments/cluster_label/results_{CORPUS}.csv")
