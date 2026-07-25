# Cluster-then-label, and a groundability predictor

Two follow-ups to the naive-fusion / hybrid-gate line of the main report:

1. **Cluster-then-label** — the "right" division of labor: let *text* induce the
   sense partition (its strength), and use the image anchors only to *name* the
   resulting clusters (their strength), instead of forcing images to improve the
   partition (which fails: §6.4, §11).
2. **Groundability predictor** — a gold-free filter that decides, per word,
   whether the image channel should be applied at all.

All over the `multi_visual` words (≥2 grounded senses) at ImageNet-21k.
Scripts: `cluster_then_label.py <corpus>`, `groundability.py`.

---

## 1. Cluster-then-label

### Variant (a): can the anchors correctly *name* a text cluster?

Cluster usages by text (spherical k-means, oracle K = gold_k), then name each
cluster by the **majority anchor-sense** of its usages; accuracy = fraction of
usages whose cluster-name equals their gold sense name. Compared to Q1's
*per-usage* anchor assignment (the noisy baseline).

**Only well-defined on SemCor.** The metric needs the gold inventory and the
anchor grounding to share sense names — true for WordNet-gold SemCor, but DWUG
(correlation clusters) and SemEval (OntoNotes / different keys) give gold names
the anchors can never match, so "naming accuracy" is undefined there (ARI in
variant (b) still works, being label-invariant).

| SemCor (64 words) | accuracy on nameable senses |
|---|--:|
| Q1 per-usage anchor labeling | 0.516 |
| **cluster-then-label** (majority vote) | 0.502 |

Flat on average (−0.015), but the average hides a concentrated win: on **11
words** the naming gain exceeds +0.05, and on the clean visual words it is
*perfect* — `head` 0.61→**1.00**, `action` 0.76→**1.00**, `point`/`table`/
`position`→**1.00**. Aggregating the cluster denoises completely where the text
cluster is pure and the senses are visually distinct. (The low ~0.20 accuracy
over *all* usages is a coverage ceiling — most usages fall on gold senses that
have no anchor and can never be named.)

**Takeaway:** images can put a correct, inspectable grounded *name* on a clean
text cluster — reliably on the visual words — far better than per-usage
assignment. But it does not raise any partition metric (naming is
partition-invariant by construction).

### Variant (b): does using the names to *merge* clusters change induction ARI?

Merge text clusters that receive the same majority anchor-name; recompute ARI.

| corpus | oracle-K text → merged | unknown-K text → merged |
|---|--:|--:|
| SemCor | 0.389 → 0.090 (**−0.30**) | 0.393 → 0.083 (−0.31) |
| DWUG EN | 0.368 → 0.264 (−0.10) | 0.475 → 0.250 (−0.22) |
| SemEval-2013 | 0.331 → 0.168 (−0.16) | 0.410 → 0.221 (−0.19) |
| SemEval-2010 | 0.616 → 0.132 (**−0.48**) | 0.591 → 0.095 (−0.50) |

**Robustly catastrophic on every corpus.** The anchor labels *collapse* (many
distinct, correct text clusters get the same dominant anchor name — the
`man`→"person" 97% collapse generalized), so merging by name fuses senses text
had correctly separated. Merge helped 0–3 words per corpus, hurt the rest.

**Conclusion.** Cluster-then-label is valuable as a *labeling / interpretability*
overlay on the visual words, but not as a way to improve WSI: naming can't change
the partition (a), and letting it change the partition is destructive (b).

---

## 2. Groundability predictor (gold-free)

Should we decide *a priori* which words the image channel can handle, and skip
the rest? §6.7 showed `multi_visual` (has an anchor) is necessary but far from
sufficient — 74/96 grounded words still fail. Test two gold-free signals against
the outcome Δ = anchor-assignment ARI − null:

| signal | what it measures | Spearman(·, Δ) | verdict |
|---|---|--:|---|
| **geometric** — mean pairwise cosine between a word's sense-anchor prototypes | are the anchor *prototypes* far apart? | **+0.08** | useless |
| **behavioral** — share of usages the assignment routes to its single most-used anchor | does the assignment *spread* across senses on real usages? | **−0.40** | usable |

**The intuitive geometric signal fails.** Words whose anchors are *most* separated
(`twist`, `means`, `foundation`, `heart`, `activity`, `thing`; pairwise cosine
0.04–0.13) mostly **fail** — abstract senses get mapped to spuriously distant but
*meaningless* anchor classes. "Are the prototypes different?" is the wrong
question.

**The behavioral signal works.** Words that work collapse onto one anchor 67% of
the time; words that fail, 87%. A gold-free rule "keep words whose assignment does
not collapse (`max share < 0.80`)":

| | words kept | precision | recall | mean Δ kept | mean Δ rejected |
|---|--:|--:|--:|--:|--:|
| `share < 0.80` | 35 / 96 | 0.43 | 0.68 | **+0.134** | **+0.010** |

Kept words average Δ **+0.134** (above the +0.10 "works" bar); rejected words
average **+0.010** (at null). A behavioral self-diagnostic — *does the visual
assignment collapse onto one sense?* — cleanly separates the words where the
image channel is worth applying from where it is not, with no gold labels.

**Contrast with the hybrid gate (§11).** That gate failed because it needed to
detect where *text* is wrong (text is confidently wrong → undetectable). This
succeeds because it asks a question about the *image* system's own behavior on the
word, which is observable a priori.

### Design implication

The honest front-end for the visual channel is **behavioral, not geometric**: run
the anchor assignment, and if it collapses onto a single sense, declare the word
non-groundable and fall back to text. This turns §6.7's post-hoc works/fails split
into an a-priori, gold-free filter — "the detector selects ~1/3 of grounded words,
and on those the image channel beats null; on the rest it correctly declines."

## Reproduce
    # per corpus: regenerate targets + caches (embed on GPU), then:
    .venv/bin/python experiments/cluster_label/cluster_then_label.py <corpus>
    # groundability (needs cache/img_bank.pt + per-corpus targets):
    .venv/bin/python experiments/cluster_label/groundability.py
