# Seeing Word Senses: Visual Anchors for Word-Sense Induction

*The complete report of the `vision-lsi` pilot. Self-contained: no prior
exposure to this repository is assumed, and the core concepts are introduced
as they are needed.*

---

## 1. Executive summary

**The question.** When the word *plane* appears in a sentence, does it mean the
aircraft or the carpenter's tool? Grouping the usages of an ambiguous word into
its senses, without a predefined sense list, is called **word-sense induction
(WSI)**. It is normally done by clustering text embeddings. This pilot asks
whether **images** can contribute: for each usage of a word, we measure how
similar its meaning is to a panel of candidate *pictures* (ImageNet class
prototypes), and test what that visual similarity buys.

**The thesis.** This work is an *introduction to the idea* of multimodal WSI,
not a claim that images beat text. The interesting property of an image anchor
is that it is **concrete, visible, and nameable**: unlike an anonymous text
cluster, an ImageNet class *is* an inspectable set of pictures with a name, so
it can serve as a grounded sense label that any new usage can be assigned to
with one dot product — no clustering, no choosing a number of clusters.

**The answers**, over four English WSI benchmarks and one multimodal encoder:

1. **Can visible anchors label usages by sense? Yes.** Assigning each usage to
   its nearest grounded anchor beats a shuffled-image control on **all four
   corpora** once the visual inventory is large enough (ImageNet-21k), and the
   effect is statistically significant on two (SemCor: Δ = +0.046,
   95% CI [+0.010, +0.084] over 64 words; SemEval-2013: Δ = +0.164,
   [+0.026, +0.339] over 6 words). A handful of words (`head`, `cell`, `part`,
   `bit`, `level`) even beat text-only clustering.
2. **Is the visual signal real? Yes, and selective.** For visually polysemous
   words the anchor similarities alone recover senses far above chance (`head`
   0.74, `cell` 0.77 ARI); for abstract words they sit at chance — the method
   does not hallucinate signal where there is none.
3. **Does it beat a strong text model under naive fusion? No**, consistently,
   under both inventories — and follow-up experiments pin the failure on the
   *fusion mechanism* and on which axis each modality splits, not on a lack of
   visual coverage.

**The constructive endpoint.** The effect is strongly word-dependent (22 of 96
testable words clearly work), so the practical question became: *can we decide,
without gold labels, which words to trust the image channel on?* Three
gold-free gate signals were tested. Text-confidence gating fails (text is
confidently wrong, not unsure). Gating on whether the image assignment
collapses onto one sense works modestly. Gating on whether the image assignment
**agrees with the text clustering** works best — a combined gate keeps 17 of 96
words at precision 0.71 and mean effect +0.26. But no gold-free signal can find
the words where the image *beats* text. The honest role of the image channel is
therefore a **certified grounding and naming layer** on top of text clustering,
not a text-fixer — and on that role the evidence is positive.

---

## 2. The idea

### 2.1 Word-sense induction, briefly

A polysemous word like *bat* has usages that split into senses (the animal, the
sports equipment). WSI is the **unsupervised** task of recovering that split:
given many sentences containing the word, partition them by sense, with no
sense inventory given in advance. The standard approach is *distributional*:
embed each usage in context with a language model and cluster the vectors.

### 2.2 Why images might help

Many sense distinctions are distinctions between **visible kinds of thing**:
*crane* (bird vs. machine), *plane* (aircraft vs. carpentry tool), *cell*
(biological vs. prison vs. phone). If we can measure, for a given usage, how
much its meaning "looks like" each of several candidate visual concepts, that
similarity vector is a perceptual fingerprint that might sharpen sense
boundaries text alone blurs.

### 2.3 What images offer even if they never beat text

A text clustering outputs anonymous groups ("cluster 3"). An image anchor
outputs a **named, inspectable** label ("this usage is the
`airliner`-like sense — here are its pictures"). And because the label is
assigned by a single similarity comparison, it is **inductive**: any new,
unseen usage can be labeled the same way, with no re-clustering and no need to
know the number of senses. Those properties — grounding, naming, induction —
are the thesis; beating text on clustering metrics is the stretch goal.

### 2.4 Three ingredients make this testable

1. **ImageNet classes are WordNet synsets.** WordNet is a lexical database
   whose word senses (synsets) form a hierarchy; ImageNet's classes are leaf
   synsets of it. So from any word we can enumerate candidate visual concepts
   through WordNet — a principled bridge from a word to a set of images.
2. **A shared text–image embedding space.** We use one multimodal encoder
   (`Qwen/Qwen3-VL-Embedding-8B`) that embeds both sentences and images into a
   single 4096-dimensional space, so `cos(text, image)` is meaningful.
3. **The anchor profile.** For usage *i*, define `a_i[c] = cos(t_i, v_c)` over
   the word's candidate visual classes *c*, where `t_i` is the usage's text
   embedding and `v_c` is class *c*'s image prototype (an average of its
   images' embeddings).

---

## 3. How to read the numbers

- **ARI (adjusted Rand index)** is the primary metric: it compares a predicted
  partition to the gold partition, is 1.0 for a perfect match, and is
  **chance-corrected** — random labeling scores 0 in expectation, and worse-
  than-chance partitions go negative. So `ARI > 0` already means real
  structure. (V-measure and B-cubed F1 are reported in the run artifacts but
  ARI carries the conclusions.)
- **Every visual system is compared to a permuted null.** The null keeps the
  system identical — same dimensionality, same scale, same decision rule — but
  scrambles *which images belong to which class*. This controls for the
  possibility that "more features help k-means" or "any anchor-shaped feature
  helps," regardless of whether the images are correctly bound. The quantity
  that matters is always the **difference Δ** between a system and its null.
- **Uncertainty** is quantified with a paired bootstrap over words (resampling
  words, keeping each word's system-minus-null difference paired), reported as
  a 95% confidence interval.
- **One caveat, found in a later audit** (§11.2): on a few words the shuffled
  null itself lands *high* (up to 0.43), because shuffled anchors also produce
  degenerate partitions that can accidentally align with a skewed gold
  distribution. Corpus-level aggregates are unaffected, but strongly *negative*
  per-word Δs should be read as "the null got lucky," not "images actively
  mislead." Flagged where it matters (§9).

---

## 4. Data: four English WSI benchmarks

Each corpus provides (target word, usage in context, gold sense). Gold labels
differ in origin but are treated identically downstream — as opaque per-word
cluster labels used **only for scoring**, never as input. All corpora are
**nouns only** in this pilot.

| corpus | gold senses | inventory | notes |
|---|---|---|---|
| **SemCor** | WordNet synsets | dependent | NLTK; broad-coverage sense-tagged corpus |
| **DWUG EN** v3.0.0 | correlation clusters over human relatedness judgments | **free** | diachronic; two time periods **pooled** (sense induction, not change detection); `-1` noise cluster dropped |
| **SemEval-2013 Task 13** | WordNet-3.1 sense keys | dependent | *graded* (multi-sense) labels; single-sense subset used for hard clustering |
| **SemEval-2010 Task 14** | OntoNotes-style sense ids | dependent | no target-token offset in the data → target located within the sentence |

DWUG EN matters most in principle: its senses are **inventory-free** (induced
from pairwise human judgments), which is the honest setting for sense
*induction*. The others ensure conclusions do not rest on one dataset's
idiosyncrasies.

Each dataset has a dedicated extractor emitting a common occurrence schema, so
every downstream stage is identical regardless of corpus.

---

## 5. Method

### 5.1 Selecting target words and grounding their senses

We keep words with ≥ 40 occurrences, whose senses each have ≥ 8 examples, and
which retain ≥ 2 such senses. (The per-sense threshold also prunes DWUG's long
tail of singleton clusters.)

**Visual grounding.** ImageNet classes are *specific leaf* synsets
(`airliner`, `soccer_ball`) while a word's senses are higher-level (`airplane`,
`ball`), so a direct lemma match finds almost nothing. Instead we expand down
the WordNet hierarchy: an ImageNet class grounds a word sense if it lies within
**3 hypernym levels below it** (`airliner` → `airplane`), capped at 12 anchor
classes per sense. A word's anchor set is the union of its grounded senses'
classes.

Crucially, grounding uses the **WordNet senses of the word**, *independent* of
the corpus gold labels: the anchors are candidate concepts, never supervision.

Words are stratified by their number of visually grounded senses `g_w`:

- `multi_visual` (`g_w ≥ 2`) — at least two senses with distinct anchors; the
  only words where anchor *assignment* can be tested;
- `visual_nonvisual` (`g_w = 1`) — one concrete sense vs. everything else;
- `text_only` (`g_w = 0`) — excluded from visual comparisons.

### 5.2 One encoder for both modalities

`Qwen/Qwen3-VL-Embedding-8B` (bf16, 4096-d, last-token pooling,
instruction-aware). One model produces everything:

- **Usage text embedding** `t_i`: the ±20-word window around the target,
  encoded with a **target-aware instruction** ("Represent the meaning of the
  word *{lemma}* as used in the following context."). The same vector is reused
  as the clustering base (after a global PCA to 64 dimensions) and, raw, as the
  cross-modal anchor query.
- **Image prototype** `v_c`: 32 sampled training images of class *c*, embedded,
  mean-pooled, L2-normalised. These are **whole images, no cropping** — a
  choice that turned out to matter (§10).
- **Label prototype**: the text embedding of the class *name* — a control that
  separates "the picture helps" from "the name of the concept helps."

### 5.3 Two visual inventories

- **ImageNet-1k** (ILSVRC-2012): 1,000 curated classes, ~1,300 images each.
- **ImageNet-21k, targeted**: rather than the full ~1.3 TB release, we
  downloaded **only the 1,115 anchor classes the four corpora actually need**
  (~81 GB; 216 further wanted classes are absent from the winter21 release) and
  merged them with ILSVRC into a 2,115-class root.

Coverage is the central practical constraint: few polysemous nouns have ≥ 2
senses that each land under an ImageNet-1k subtree. The targeted 21k expansion
lifts the assignment-testable vocabulary **from 12 to 96 words** (~8×):

| corpus | words selected | visual (1k → 21k) | multi_visual (1k → 21k) |
|---|---|---|---|
| SemCor | 209 | 25 → 99 | 7 → **64** |
| DWUG EN | 21 | 6 → 15 | 3 → **12** |
| SemEval-2010 | 37 | 4 → 20 | 2 → **14** |
| SemEval-2013 | 15 | 3 → 10 | 0 → **6** |

### 5.4 Systems compared

| system | features | role |
|---|---|---|
| `qwen` | PCA-64 of `t_i` | text-only baseline |
| `qwen+image` | `[h̃ ; λ·zscore(a_i)]`, `a_i[c]=cos(t_i,v_c)` | fusion hypothesis |
| `qwen+label` | same, `v_c` = class-*name* text embedding | name-vs-image control |
| `qwen+shuffled-image` | same, class→prototype identity permuted | fusion null |
| `image-profile-only` | `a_i` alone (k-means) | is the anchor informative by itself? |
| `shuffled-profile-only` | permuted `a_i` alone | null for the line above |
| `anchor-assignment` | `argmax_s max_c cos(t_i,v_c)` (no k-means) | **headline: nearest-visible-anchor labeling** |
| `anchor-assignment-shuffled` | same, permuted prototypes | null for assignment |

`anchor-assignment` assigns each usage directly to its nearest *grounded
sense*, pooling the ImageNet classes under each sense (so a sense with several
hyponym classes is not over-split). It uses no clustering and no K, so it
applies to arbitrary unseen usages and its induced senses are *named* by their
ImageNet classes.

Fusion vectors are L2-normalised; the fusion weight λ is chosen per word by
leave-one-word-out tuning (each word gets the λ that is best *on the other
words*), so no word tunes on its own gold labels.

### 5.5 Clustering and evaluation

Spherical k-means per word, 10 seeds, two regimes: **oracle-K** (gold number of
senses supplied — an upper-bound setting) and **unknown-K** (K selected by
cosine silhouette — the realistic setting). Sixteen full runs share one
pipeline: 4 corpora × {oracle, unknown} × {1k, 21k}.

Three pre-registered comparisons, in order of importance to the thesis:

- **Assignment test** (headline): `ARI(anchor-assignment) − ARI(its null)`,
  scored on `multi_visual` words (words with one grounded sense produce one
  assignment cluster and sit at ARI 0 by construction — they would only dilute
  the mean).
- **Signal test**: `ARI(image-profile-only) − ARI(shuffled-profile-only)` —
  does the anchor profile *alone* carry class-specific sense information?
- **Fusion test** (the hard, secondary bar): `ARI(qwen+image) − ARI(qwen)`,
  with the label and shuffled controls and a strict 6-criterion go/no-go rule.

Hardware: one NVIDIA RTX PRO 6000 (Blackwell); embeddings in bf16.

---

## 6. Result 1 — nearest-visible-anchor assignment works (headline)

Label each usage by its nearest grounded-sense anchor and score the induced
partition. Assignment is K-free, so oracle/unknown-K are identical here.
Macro-ARI over `multi_visual` words, under both inventories:

| corpus | 1k: #words, assign / null | 21k: #words, assign / null |
|---|---|---|
| SemCor | 7 — 0.058 / 0.045 | **64 — 0.080 / 0.034** |
| DWUG EN | 3 — 0.245 / 0.034 | 12 — 0.199 / 0.106 |
| SemEval-2013 | 0 — n/a | **6 — 0.208 / 0.043** |
| SemEval-2010 | 2 — −0.032 / 0.018 | 14 — 0.103 / 0.084 |

Under ImageNet-1k the evidence rested on a couple of words (`plane` ARI 0.70
vs. 0.02 null was the original anecdote). Under the 21k expansion assignment
beats its null **on all four corpora**, and the paired bootstrap gives the
study's first significant aggregates:

| corpus (21k) | n words | Δ (assignment − null) | 95% CI | excludes 0 |
|---|---:|---:|---|---|
| **SemCor** | 64 | **+0.046** | [+0.010, +0.084] | **yes** |
| **SemEval-2013** | 6 | **+0.164** | [+0.026, +0.339] | **yes** |
| DWUG EN | 12 | +0.093 | [−0.017, +0.221] | no (small n) |
| SemEval-2010 | 14 | +0.019 | [−0.058, +0.115] | no |

(Scored over *all* visual words — including the single-grounded-sense zeros —
the SemCor figure is a diluted but still-significant +0.030 [+0.007, +0.056].)

The effect is no longer a two-word anecdote. Words where assignment beats its
null by a wide margin now include:

| word | assignment ARI | null | text (`qwen`) |
|---|---:|---:|---:|
| `cell` (SemEval-2010) | 0.93 | 0.43 | 1.00 |
| **`plane`** (DWUG) | 0.68 | 0.18 | 0.72 |
| **`head`** (DWUG) | **0.68** | 0.17 | **0.32** |
| `plant` (SemCor) | 0.67 | 0.07 | 0.84 |
| `course` (SemCor) | 0.65 | 0.08 | 0.78 |
| **`cell`** (SemCor) | **0.58** | 0.07 | **0.33** |
| `paper` (SemEval-2013) | 0.59 | 0.06 | 0.67 |
| **`part`** (SemEval-2013) | **0.43** | 0.13 | **0.11** |
| `floor` (SemCor) | 0.36 | 0.07 | 0.92 |

Bolded rows are cases where the visual assignment **outperforms text-only
clustering** (`head`, `cell`, `part`; also `bit`, `level`) — the first direct
instances of the visual channel beating the text baseline on any measure.

What the rule does, qualitatively — usages of `plane` and `cell` grouped by
their nearest ImageNet anchor image, with no sense labels and no text
supervision:

![One word, several pictures: usages of "plane" and "cell" grouped by their nearest ImageNet anchor image](figure_word_senses.png)

Three takeaways. **(a)** Nearest-visible-anchor assignment — a named,
inspectable, inductive sense labeling with no clustering — works, and with
adequate coverage the effect is corpus-level, not anecdotal. **(b)** It remains
strongly word-dependent (§9): the aggregate is carried by genuinely visual
words. **(c)** For several words the signal is present in the profile
*geometry* yet raw argmax underperforms it (§7) — the anchor space is
informative but the nearest-anchor decision rule is uncalibrated.

---

## 7. Result 2 — the signal is real, but k-means on the profile wastes it

The second question: does the anchor profile *alone* — the vector of
similarities to the candidate classes, with the text clustering base removed —
contain sense information? Cluster it with k-means and compare to its permuted
null.

**Aggregate: at or near chance.** Under 21k the deltas are SemCor
+0.020/+0.014, SemEval-2013 +0.063/+0.051, DWUG +0.006/−0.034, SemEval-2010
+0.010/−0.014 (oracle/unknown-K). Notably, the 21k expansion did **not** rescue
this test the way it rescued assignment: absolute profile ARIs rose (SemCor
0.07 → 0.19) but the *null* rose in lockstep — more anchor dimensions help
k-means regardless of whether the images are correctly bound. That contrast is
informative: reading a single nearest-anchor label off the profile exploits
class-specific structure that whole-vector clustering blurs.

**Per word: sharp, replicated structure.** Under 21k, ~22 `multi_visual` words
cluster their profile above null (Δ > 0.10, oracle-K), and the strong cases
replicate across clustering regimes (oracle / unknown-K):

| word (corpus) | profile-only ARI | profile-null |
|---|---|---|
| **`head`** (SemCor) | **0.74 / 0.74** | 0.28 / 0.45 |
| **`cell`** (SemCor) | **0.77 / 0.77** | 0.55 / 0.54 |
| **`face`** (DWUG) | **0.41 / 0.41** | 0.07 / 0.07 |

(others: `paper`, `officer`, `section`, `grain`, `body`, `house`, `field`, …).
The clustering "winners" differ from the assignment winners — `head`/`face`
cluster strongly while `plane`/`cell` *assign* strongly — because argmax and
whole-profile k-means read the same profile differently.

**And it is inert where it should be.** Abstract nouns sit at chance (`market`
0.00; `area`, `church`, `door` ≈ 0), and some nominally-visual words whose
senses are not visually separable fall at or below null. This selectivity —
strong where senses are visually distinct, null where they are not — is the
core evidence that the visual channel encodes real, interpretable sense
information rather than a generic extra-features artifact.

---

## 8. Result 3 — naive fusion does not beat a strong text model

The hard bar: concatenate the anchor profile onto the text embedding
(`[h̃ ; λ·zscore(a_i)]`) and cluster. The ImageNet-1k results, all corpora and
both K regimes (`profile` columns are §7's test; `Δ_image = +img − text` is
this one):

| corpus (mode) | #vis | profile | profile-null | Δ_signal | text | +img | Δ_image | +lbl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SemCor (oracle) | 25 | 0.069 | 0.065 | +0.004 | 0.402 | 0.320 | −0.082 | 0.298 |
| SemCor (unknown) | 25 | 0.065 | 0.069 | −0.004 | 0.385 | 0.350 | −0.034 | 0.324 |
| DWUG EN (oracle) | 6 | 0.156 | 0.150 | +0.006 | 0.554 | 0.545 | −0.009 | 0.504 |
| DWUG EN (unknown) | 6 | 0.215 | 0.126 | +0.089 | 0.705 | 0.696 | −0.008 | 0.648 |
| SemEval-2013 (oracle) | 3 | 0.028 | −0.001 | +0.029 | 0.364 | 0.354 | −0.010 | 0.418 |
| SemEval-2013 (unknown) | 3 | 0.031 | −0.002 | +0.033 | 0.390 | 0.394 | +0.004 | 0.433 |
| SemEval-2010 (oracle) | 4 | 0.036 | 0.053 | −0.018 | 0.550 | 0.537 | −0.013 | 0.512 |
| SemEval-2010 (unknown) | 4 | 0.067 | 0.073 | −0.006 | 0.558 | 0.541 | −0.017 | 0.530 |

`Δ_image ≤ 0` in essentially every run under **both** inventories; under 21k it
is negative on all four corpora (−0.04 to −0.09). Two qualifications:

- **The target-aware instruction dramatically strengthened the text baseline**
  (DWUG `qwen` rose from ~0.24 with a plain instruction to 0.55/0.71). This is
  the single largest effect in the whole study — and it is on the *text* side.
- More (and more correctly-bound) anchors did not help fusion. The same signal
  that yields a significant *assignment* result is wasted under fixed-λ
  concatenation — the failure is in the mechanism, not the coverage.

### 8.1 Ruling out a scaling artifact

To check the failure is not a dimensionality/scaling accident, we swept how the
visual block is built before concatenation (SemCor @21k, multi_visual,
oracle-K) — the profile `a_i`, and the *selected prototype* `v_{c*}` (the
argmax anchor's full 4096-d vector), hard and softmax-weighted, at three
scalings, each with per-word λ tuning:

| visual block | raw | PCA-32 img | full-4096 both |
|---|--:|--:|--:|
| profile `a_i` (= `qwen+image`) | 0.301 | — | — |
| selected prototype, argmax | 0.155 | 0.231 | 0.161 |
| selected prototype, softmax | 0.189 | 0.230 | 0.191 |
| **text baseline** | **0.389** (PCA-64) / **0.399** (full-4096) | | |

Every variant loses to text, and the tuned λ **pins at the grid floor in every
case** — the optimiser wants to *ignore* the visual block. Two structural
reasons, not scale: (i) the selected-prototype feature is constant within each
argmax group, so concatenating it drags the strong text clustering (0.39)
toward the much weaker assignment partition (~0.08); (ii) a z-scored block of
`d` dimensions has norm ∝ √d, which swamps the unit-norm text vector unless
λ→0. (Incidental notes: full-4096 text ≈ PCA-64 text, so PCA costs nothing;
softmax pooling consistently beats hard argmax.)

### 8.2 Image vs. name

On SemEval-2013 the class **name** helps fusion more than the class *image*
(`+lbl` ≈ 0.42–0.43 vs `+img` ≈ 0.35–0.39, and above text) — a reminder that
lexical grounding and visual grounding are different signals, and that for some
words the *name* of the visual concept is the more useful cue. §13 returns to
this at the assignment level, where the comparison flips.

---

## 9. Which words does it work for? — the full inventory

The effect is strongly word-dependent, so here is the whole picture. Of the 96
assignment-testable (`multi_visual`) word–corpus pairs under ImageNet-21k,
classified by Δ = assignment − null with a conservative cut (a word only counts
as working when it *clearly* beats its null):

| outcome | Δ | count | share |
|---|---|--:|--:|
| **clearly works** | ≥ +0.10 | **22** | 23 % |
| marginal | +0.02 to +0.10 | 14 | 15 % |
| **no signal** | < +0.02 | 60 | 62 % |

The works-rate is broadly similar across corpora (SemCor 13/64, DWUG 4/12,
SemEval-2013 3/6, SemEval-2010 2/14).

### The words that work (Δ ≥ +0.10)

| word (corpus) | assign | null | Δ | anchored sense split |
|---|--:|--:|--:|---|
| `plant` (SemCor) | 0.67 | 0.07 | +0.60 | flora vs. factory |
| `course` (SemCor) | 0.65 | 0.08 | +0.58 | class / racecourse / path |
| `paper` (SemEval-13) | 0.59 | 0.06 | +0.53 | newspaper vs. sheet of paper |
| `head` (DWUG) | 0.68 | 0.17 | +0.51 | body-part / promontory / drum-head / tape head |
| `cell` (SemCor) | 0.58 | 0.07 | +0.51 | biological / prison / phone |
| `cell` (SemEval-10) | 0.93 | 0.43 | +0.50 | biological / prison / phone |
| `plane` (DWUG) | 0.68 | 0.18 | +0.50 | aircraft vs. carpenter's plane |
| `head` (SemCor) | 0.36 | 0.04 | +0.32 | as above |
| `ground` (SemCor) | 0.39 | 0.09 | +0.30 | earth / land / undercoat |
| `part` (SemEval-13) | 0.43 | 0.13 | +0.29 | portion vs. region |
| `floor` (SemCor) | 0.36 | 0.07 | +0.29 | storey / ground / sea-floor |
| `body` (SemCor) | 0.35 | 0.08 | +0.27 | organism / torso / corpus |
| `officer` (SemEval-10) | 0.32 | 0.09 | +0.24 | military / police / official |
| `bit` (DWUG) | 0.26 | 0.04 | +0.22 | drill bit vs. horse's bit |
| `officer` (SemCor) | 0.26 | 0.05 | +0.21 | as above |
| `level` (SemCor) | 0.21 | 0.00 | +0.21 | storey vs. horizontal surface |
| `control` (SemEval-13) | 0.18 | −0.00 | +0.18 | control panel vs. restraint |
| `section` (SemCor) | 0.27 | 0.11 | +0.16 | segment vs. district |
| `bar` (DWUG) | 0.40 | 0.25 | +0.15 | rod / pub / rifle |
| `paper` (SemCor) | 0.16 | 0.04 | +0.12 | newspaper vs. sheet |
| `center` (SemCor) | 0.22 | 0.10 | +0.12 | middle vs. facility |
| `man` (SemCor) | 0.10 | −0.01 | +0.11 | human / manservant / soldier (barely) |

The 14 marginal words include `water`, `face`, `community`, `element`, `part`
(DWUG), `picture`, `space`, `side`, `book` — a positive nudge, not enough to
rely on.

### The words that fail, and why

Three failure modes recur:

- **(a) The senses look alike.** `man`'s senses are all people (its usages
  route to a single generic-person anchor **97%** of the time), `board`'s are
  all flat panels, `house`/`body`'s are all buildings or bodies; `child`'s and
  `girl`'s anchor prototypes are literally *identical* (cosine 1.00).
- **(b) A physical word whose sense split is abstract.** `yard` (the length
  unit), `film` (the movie), `ball` (the dance) — the distinguishing sense is
  not a picturable object.
- **(c) Text already saturates.** `yard`, `board`, `body` have text ARI ≈ 1.00
  — no headroom for anything to help.

A note on the most negative rows (`house` −0.25, `board` −0.24, `ball` −0.23):
a later audit (§11.2) showed these words' shuffled **nulls are inflated**
(0.20–0.29) — shuffled anchors also collapse to a single sense, and a collapsed
partition can accidentally align with a skewed gold distribution. So these rows
mean "no usable signal," not "images actively mislead."

Conversely, the words that work share one property: **their senses denote
different kinds of object** (flora/factory, aircraft/tool, cell/prison/phone).
That — not lemma concreteness — is the operative condition: several concrete
nouns fail, and even words with *every* gold sense anchored are mixed (`plant`,
`floor`, `head`, `officer` work; `yard`, `film`, `book`, `child`, `girl` do
not). The same lemma can also land differently in different corpora (`body`,
`part`, `officer`), because corpora attest different sense mixes.

---

## 10. What does the prototype actually see? — the cropping A/B

The image prototypes are whole images, and on 1,200 annotated ILSVRC images the
object's bounding box covers a **median of only ~45%** of the frame (66% of
images < 60%; 46% < 40%). The obvious hypothesis: the background is noise, and
cropping to the object would sharpen the anchor. We tested it directly on
SemCor @1k — same 32 annotated images per anchor class, two matched prototypes
(whole image vs. cropped to the object bounding box, which removes ~48% of the
frame), everything else identical.

**Cropping hurt, sharply** (multi_visual assignment, oracle-K):

| prototype variant | assign | null | Δ |
|---|---:|---:|---:|
| whole image | 0.072 | 0.042 | **+0.030** |
| cropped to bbox | 0.001 | 0.020 | **−0.019** |

The two words carrying the whole-image signal lost it entirely: `board`
0.378 → 0.000, `light` 0.120 → 0.005.

The likely reading flips the intuition: **the surrounding scene carries
sense-relevant signal.** `board`'s anchors are `plank` vs. `dining_table` — as
whole images they differ enormously by *scene* (workshop vs. dining room), and
the scene correlates with the sense; cropped to the bare object, two flat
wooden surfaces look *more* alike. Caveats: small n (only two words had signal
to lose), and crop-and-upscale degrades the image, which is confounded with
context removal — the clean follow-up is masking/blurring the background *in
place*. (The one-off script for this A/B was not retained; its outputs are
archived in `results/semcor_1k_{whole,crop}bbox_oracle_k/` and
`cache/proto_{whole,crop}.pt`. The main pipeline is untouched.)

---

## 11. Making it usable: when should the image channel be trusted?

Only ~23% of grounded words clearly work (§9), so the practical question is
whether we can decide **per word, without gold labels**, when to apply the
image channel. Three gate signals were tested; the third works. This section
also contains the audits that qualify the second.

### 11.1 Gating on text uncertainty — fails, instructively

The natural hybrid: keep text clustering, fall back to image assignment on the
words text can't handle, detected by a gold-free text-uncertainty signal (how
much the text partition wobbles across the 10 k-means seeds). Over the
`multi_visual` words at 21k:

| corpus | n | text | oracle ceiling | best realistic gate | corr(uncertainty, image−text) |
|---|--:|--:|--:|--:|--:|
| SemCor | 64 | 0.389 | 0.402 (+0.013) | 0.389 (+0.000) | +0.24 |
| DWUG EN | 12 | 0.368 | 0.423 (+0.055) | 0.398 (+0.030) | +0.19 |
| SemEval-2013 | 6 | 0.331 | 0.383 (+0.052) | 0.331 (+0.000) | −0.61 |
| SemEval-2010 | 14 | 0.616 | 0.618 (+0.002) | 0.616 (+0.000) | +0.35 |

Two findings, both negative:

1. **The ceiling is tiny.** Even an *oracle* that picks the better system per
   word gains only +0.019 ARI pooled ("always use image" is catastrophic:
   SemCor 0.389 → 0.080). Realistic gates recover ~+0.004.
2. **The premise is false.** The words image rescues are text's most *stable*
   words (in SemCor, the rescued words sit at uncertainty ranks 46–56 of 64).
   Text is not unsure about them — it is **confidently wrong**, partitioning
   cleanly by topic/register instead of by sense. Any text-internal confidence
   measure (seed stability, silhouette, margin) rewards well-separated
   clusters, and these clusters *are* well separated — on the wrong axis.
   (DWUG's +0.030 rides one word, `head`, that happens to be both the top
   rescue and the most text-uncertain word; the correlations are weak and flip
   sign on SemEval-2013.)

### 11.2 Gating on the image system's own behavior — works modestly, with a caveat

Test two gold-free signals against the outcome Δ = assignment − null, over the
96 testable words:

| signal | what it asks | Spearman(·, Δ) | |
|---|---|--:|:--|
| **geometric** — mean pairwise cosine between a word's sense-anchor prototypes | are the prototypes far apart? | +0.08 | useless |
| **behavioral** — share of usages routed to the single most-used anchor | does the assignment collapse onto one sense? | −0.40 | usable |

The intuitive geometric signal fails: the words with the *most* separated
anchors (`twist`, `means`, `foundation`, `heart`, `activity`, `thing`) mostly
fail — abstract senses map to spuriously distant but *meaningless* anchors.
The behavioral signal works as a filter: keeping only words whose assignment
does not collapse (`max share < 0.80`) keeps 35/96 words at mean Δ **+0.134**
(above the +0.10 "works" bar) while the rejected words sit at **+0.010** —
precision 0.43, recall 0.68.

**Audit — the behavioral signal is partly mechanical.** A collapsed assignment
is a near-one-cluster partition, and such partitions have ARI ≈ 0 *by
construction* — so "collapse predicts failure" is partly tautological rather
than a measurement of visual separability. Decomposed: of the 61 collapsed
words only 10 have assignment ARI > 0.05 at all, and *within* the 35
non-collapsed words the correlation drops to Spearman −0.27 (works-rate 0.43).
The filter remains operationally valid — degenerate output is useless whatever
its cause — but its diagnostic content is thinner than it first appears. The
same audit surfaced the **inflated nulls** noted in §3/§9: shuffled prototypes
also collapse, and on 8 words the null lands at 0.15–0.43 (`cell`@SemEval-2010
0.43, `house` 0.29, `board` 0.28, `ball` 0.20).

### 11.3 Gating on text–image agreement — the strongest signal

A third gold-free signal uses an artifact the pipeline already has: the **ARI
between the text partition and the image assignment partition** (mean over text
seeds) — no gold involved. Since text is right on most words, agreeing with
text is a strong proxy for being right; the text clustering acts as a
*pseudo-gold* that certifies the image channel word by word.

| gold-free signal | Spearman(·, Δ) |
|---|--:|
| geometric (anchor separation) | +0.08 |
| behavioral (collapse) | −0.40 |
| **agreement** ARI(text, assignment) | **+0.53** |

Stacking agreement on the behavioral filter roughly doubles precision and
effect size:

| gate | keep | precision | recall | mean Δ kept | mean Δ rejected |
|---|--:|--:|--:|--:|--:|
| behavioral (`share < 0.80`) | 35/96 | 0.43 | 0.68 | +0.134 | +0.010 |
| **+ agreement (`> 0.10`)** | 17/96 | **0.71** | 0.55 | **+0.260** | +0.011 |

### 11.4 The limit: certification is possible, rescue detection is not

What the agreement gate *cannot* do is find the **rescue words** — the 13 words
where image assignment beats text clustering. It captures only **2 of 13**, and
the quadrant decomposition shows this is structural, not a threshold problem:

| quadrant | n | mean Δ | works (Δ≥0.10) | rescues (img>text) |
|---|--:|--:|--:|--:|
| collapsed (`share ≥ 0.80`) | 61 | +0.01 | 7 | 7 |
| spread + agrees with text | 17 | **+0.26** | 12 | **2** |
| spread + disagrees with text | 18 | +0.01 | 3 | 4 |

A word where image beats text is a word where text is *wrong* — so agreement
with text is low there by construction. And raw *dis*agreement doesn't identify
them either: the disagree quadrant averages Δ ≈ 0, because most disagreement is
the image splitting a wrong axis. Combined with §11.1, this closes the question
from all sides: **text's failures are invisible to text-internal confidence, to
the image system's own behavior, and to cross-system disagreement.** Gold-free
*certification* of the visual channel is possible (11.3); gold-free *rescue
detection* is not. The image channel's defensible unsupervised role is a
**certified grounding and naming layer**, not a text-fixer.

---

## 12. Using the anchors for names, not partitions — cluster-then-label

If the channel's role is naming, the right division of labor is: let *text*
induce the partition (its strength), and use the anchors only to **name** the
resulting clusters. Two variants, over `multi_visual` words at 21k.

**(a) Naming a text cluster works — on the visual words, perfectly.** Cluster
by text (oracle-K), name each cluster by the majority anchor-sense of its
usages, and measure naming accuracy against gold sense names (well-defined on
SemCor, whose gold shares WordNet's names). Per-usage anchor labeling scores
0.516 on nameable senses; majority-vote cluster naming scores 0.502 on average
— flat — but the average hides a concentrated win: on 11 words the gain
exceeds +0.05, and on the clean visual words naming is *perfect* (`head`
0.61→**1.00**, `action` 0.76→**1.00**, `point`/`table`/`position`→**1.00**).
Aggregating over a pure text cluster denoises the anchor signal completely.

**(b) Letting the names change the partition is destructive.** Merging text
clusters that receive the same anchor name collapses senses text had correctly
separated (the `man`→"person" 97% collapse, generalized):

| corpus | oracle-K text → merged | unknown-K text → merged |
|---|--:|--:|
| SemCor | 0.389 → 0.090 | 0.393 → 0.083 |
| DWUG EN | 0.368 → 0.264 | 0.475 → 0.250 |
| SemEval-2013 | 0.331 → 0.168 | 0.410 → 0.221 |
| SemEval-2010 | 0.616 → 0.132 | 0.591 → 0.095 |

**Conclusion.** Cluster-then-label is valuable as a labeling/interpretability
overlay on the gated visual words — exactly the role §11 certifies — but not as
a way to improve the partition itself.

---

## 13. Assignment-level refinements: the name control, calibration, and score fusion

Three follow-ups at the assignment level (currently on SemEval-2010 @21k, 14
testable words — the corpus whose features were cached; text baseline 0.616,
image-assignment null ≈ 0.084):

| assignment variant | mean ARI |
|---|--:|
| image prototypes (the pipeline system) | 0.107 |
| class-**name** (label) prototypes | 0.079 |
| image, per-sense z-calibrated | 0.082 |
| label, per-sense z-calibrated | 0.142 |
| **0.5·image + 0.5·label score fusion** | **0.148** |

- **The image content matters.** The headline (§6) never controlled for whether
  the *picture* helps or just the WordNet class *name*. It does: image
  prototypes beat name prototypes 0.107 vs 0.079 on raw argmax. (Replicating
  this control on SemCor, the headline corpus, needs one re-embed.)
- **Per-sense calibration is a trade, not a win.** Z-scoring each sense's score
  column before argmax rescues collapse-prone words (`body` 0.01→0.32) but
  destroys the strong ones (`cell` 0.93→0.24). Natural refinement: trigger
  calibration only when the raw assignment collapses — §11.2's max-share
  statistic is exactly that trigger.
- **Score-level image+name fusion is a near-free upgrade.** Averaging the image
  and label sense-scores before argmax beats both channels (0.148), keeps
  `cell` at 0.93, and lifts `body` from ~0 to **0.62**. Fusion *of decisions*
  succeeds where fusion *of features* (§8) robustly fails. (No matched shuffled
  null was run for the fused scores yet.)

---

## 14. A different grounding route: gloss anchors (DWUG prototype)

The WordNet→ImageNet bridge has an alignment gap (anchors are WordNet senses,
gold may not be) and a coverage bottleneck. DWUG EN ships human-written
**definitions (glosses) per gold cluster**, enabling a cleaner route: embed the
gloss, assign each usage to its nearest gloss. Anchors are then 1:1 with the
evaluated senses — no WordNet, no images, no coverage gap.

| anchor type (21 DWUG words) | macro ARI | notes |
|---|--:|---|
| **gloss-text** (definition embedding) | **0.307** | vs. random-gloss null 0.023; 17/21 words beat null |
| gloss-**image** (top-5 retrieved class prototypes per gloss) | 0.176 | retrieval is sane but lossy |
| oracle per-word blend | 0.334 | image > text on only 5/21 words |
| (reference: WordNet→ImageNet image anchors) | 0.199 | over only 6 testable words |

Two lessons. First, **alignment + coverage is the dominant lever**: definition
anchors, which align 1:1 with gold and cover every word, far outperform the
image route in both score and breadth — though this variant is pure text
(definition-matching WSD à la Lesk), not multimodal. Second, **routing the
gloss through image space loses signal** (0.176 < 0.307): the definition
already carries the sense; passing it through image retrieval only discards
some. The images help precisely on the visually distinct words (`land`, `ball`,
`plane`, `face`, `head` — and `head` is the notable case where text glosses
fail, −0.06, but retrieved images help, +0.07), consistent with everything
above.

---

## 15. What we now know

The picture is coherent across four benchmarks and every follow-up:

1. **Visible anchors can label usages by sense, at corpus scale.** Under
   adequate coverage (21k) nearest-anchor assignment beats its null on all four
   corpora, significantly on two, and beats text-only clustering outright on a
   handful of words. Coverage was a real ceiling and was liftable cheaply
   (~81 GB of targeted classes, not 1.3 TB).
2. **The signal is real, selective, and realized by argmax more than by
   clustering.** Visually distinct senses light up; abstract words sit at
   chance; the profile geometry often contains more than the raw argmax
   extracts.
3. **The scene is part of the signal.** Cropping prototypes to the object
   *hurt* — image context appears to disambiguate, not dilute.
4. **Text is the stronger partitioner, and naive fusion cannot change that.**
   The instruction-tuned text baseline is very strong; fixed concatenation
   provably wastes the visual signal (λ pins at the floor); text and image
   disagree on *which axis* to split, and that disagreement is invisible to
   every unsupervised confidence measure.
5. **The channel can be scoped without gold labels — for certification, not
   rescue.** A behavioral filter plus a text-agreement check selects ~1/6 of
   grounded words on which the image channel is reliably above null (mean Δ
   +0.26, precision 0.71); the words where image would *beat* text cannot be
   found gold-free, from any side.
6. **The defensible product is a certified, named, inductive grounding layer**:
   gate the words (§11.3), name text clusters through the anchors (§12 — perfect
   on the clean visual words), and assign new usages inductively (§6), with
   image+name score fusion as a free upgrade (§13).

---

## 16. Limitations

- **Sample size.** SemCor@21k reaches significance (64 words); DWUG (12),
  SemEval-2010 (14), and SemEval-2013 (6) give wide CIs — their point estimates
  are suggestive, not conclusive.
- **Visual inventory — improved, not solved.** 216 wanted synsets are absent
  from winter21; 21k tail classes are noisier and smaller (some < 32 images)
  than curated ILSVRC. Coverage is broad but uneven in quality.
- **Prototypes are whole-image averages.** A prototype conflates object and
  scene; both carry signal (§10), but an averaged "typical look" remains
  coarse, and cross-modal cosine is dominated by topic over fine sense.
- **The null is imperfect on skewed words.** Shuffled anchors can collapse
  onto partitions that accidentally align with a skewed gold (8 words with null
  0.15–0.43), inflating the null and making strongly negative per-word Δs
  uninterpretable (§11.2). Aggregates are unaffected.
- **Naive fusion only.** Learned or gated fusion is untested (and §11 shows an
  unsupervised text-side gate cannot work; supervision or an image-side signal
  is required).
- **Scope.** English, nouns only. The WordNet→ImageNet bridge is
  English-specific; non-English corpora would need a different grounding route
  (the gloss route of §14 is one candidate).
- **Anchor grounding uses WordNet senses of the lemma**, which need not align
  with a corpus's gold senses (especially DWUG's induced clusters): a word can
  be `multi_visual` yet have gold senses that do not match its visual split.
- **Single-slot caches.** The pipeline overwrites `data/targets.csv` and
  `cache/*` per corpus, so cross-corpus experiments require regenerating
  caches; some follow-up analyses instead read the committed run artifacts.

---

## 17. Future work

1. **Selective calibration.** Per-sense score calibration helps exactly the
   collapse-prone words and hurts the strong ones (§13) — trigger it with the
   collapse statistic (§11.2) instead of applying it globally.
2. **Score-level fusion, properly.** The 0.5/0.5 image+name fusion (§13) needs
   its matched null and a SemCor replication; learned decision-level fusion is
   the natural extension. Feature-level learned fusion is only worth pursuing
   *with* supervision (§11.1).
3. **Better prototypes — not by cropping.** Background mask/blur in place
   (same framing/resolution) to disentangle "context helps" from "cropping
   degrades" (§10); usage-conditioned or exemplar anchors (retrieve/weight the
   class images most like the usage) rather than one scene-heavy average.
4. **Gloss→image, done right.** The gloss route wins on alignment and coverage
   (§14); generating or retrieving *usage-specific* images per gloss, rather
   than averaging retrieved class prototypes, may close its multimodal gap.
5. **Extend to verbs and other languages**, and to diachronic sense change:
   assignment is K-free and inductive, so comparing per-period anchor
   distributions on DWUG (whose periods are currently pooled) gives a *named*
   sense-change detector with no clustering — DWUG's native task.

---

## 18. Reproducing this

```bash
make setup                                   # venv + non-torch deps + NLTK corpora
uv pip install --python .venv/bin/python torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu129     # Blackwell (sm_120)

export IMAGENET_ROOT=/path/to/ILSVRC/Data/CLS-LOC        # dir whose train/ holds n######## classes
make dwug-fetch semeval2013-fetch semeval2010-fetch      # download the corpora
CORPUS=dwug_en bash run.sh                               # or semcor / semeval2013 / semeval2010
```

For the ImageNet-21k condition, only the anchor synsets a corpus needs are
required: run `select_targets` against the 21k synset list, download just those
`winter21_whole/<wnid>.tar` tarballs, merge with ILSVRC under one root, and
point `IMAGENET_ROOT` at it (~81 GB / 1,115 classes, not the full ~1.3 TB).

**Pipeline stages** (`src/`): `audit` →
`extract_{semcor,dwug,semeval2013,semeval2010}` → `index_imagenet` →
`select_targets` → `embed_imagenet` + `embed_contexts` → `construct_features`
→ `cluster` → `evaluate` → `report`. Each stage is a `make` target and a plain
`python -m src.<stage>` module; knobs live in `configs/pilot.yaml`.

The sixteen main runs are under
`results/<corpus>_{oracle,unknown}_k/report.md` (ImageNet-1k) and
`results/<corpus>_21k_{oracle,unknown}_k/report.md` (21k). `results/` is
git-ignored — the runs live in the checkout that executed the pipeline.

**Follow-up experiments** (`experiments/`, each with its own RESULTS.md or
committed CSVs):

| section | script(s) |
|---|---|
| §10 crop A/B | one-off script, not retained (outputs in `results/semcor_1k_*bbox_oracle_k/`) |
| §11.1 hybrid gate | `experiments/hybrid/hybrid_gate.py` |
| §11.2 geometric/behavioral gates | `experiments/cluster_label/groundability.py` |
| §11.2 audits, §11.3–11.4 agreement gate | `experiments/agreement_gate/agreement_gate.py` |
| §12 cluster-then-label | `experiments/cluster_label/cluster_then_label.py` |
| §13 label control / calibration / fusion | `experiments/agreement_gate/label_assignment.py` |
| §14 gloss anchors | `experiments/gloss/eval_gloss_dwug.py`, `eval_gloss_image_dwug.py` |

**Figures.** `make figures` (or `python -m experiments.figures.<name>`
individually) regenerates the poster figures into `figures/`, which is
git-ignored — regeneration needs the run outputs in `results/`, the embedding
caches in `cache/`, and (for the two photo-tile figures) ImageNet-21k on disk;
set `IMAGENET_DIR` to its train split. The four generators in
`experiments/figures/`: `figure_q1` (the §6 assignment-vs-null deltas across
all four corpora, with paired-bootstrap CIs), `figure_anchor_plane` (the
three-panel showcase of the `cell` anchor plane), `figure_plane_simple` (the
same plane as a plain scatter), and `figure_anchor_space` (usages and anchors
in one mean-centered 2-D layout). The qualitative figure embedded in §6
(`figure_word_senses.png`) is committed directly.
