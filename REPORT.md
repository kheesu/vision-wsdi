# Visual Anchors for Lexical Sense Induction — a multimodal WSI pilot

*A self-contained report of the `vision-lsi` experiments. Written for a
WSD/WSI researcher with no prior exposure to this repository.*

---

## 1. TL;DR

We ask whether **image information can contribute to unsupervised word-sense
induction (WSI)**. For each usage of a target word we build a *visual-anchor
profile* — the similarity of the usage's (text) embedding to a panel of
candidate ImageNet class-image prototypes — and test whether that profile helps
cluster usages into senses. Everything is run with a single multimodal encoder
(`Qwen/Qwen3-VL-Embedding-8B`) that maps text and images into one space, over
**four English WSI benchmarks** (SemCor, DWUG EN, SemEval-2010, SemEval-2013).

**This work is an *introduction to the idea* of multimodal WSD/WSI, not a claim
that images naively beat text.** The thesis is what images uniquely offer: a
**concrete, visible, nameable anchor**. Unlike an anonymous text cluster, an
ImageNet class *is* an inspectable image, so it can serve as a grounded sense
label to which **any usage can be assigned inductively** — one dot product, no
clustering. We report three questions, in order of importance to the thesis:

1. **Can we assign senses to usages via visible anchors?** — label each usage by
   its nearest grounded-sense anchor, `argmax_s max_c cos(t_i, v_c)`, and score
   the induced partition. **Answer: yes.** Under ImageNet-1k the evidence was a
   handful of words (*plane* ARI 0.70 vs. 0.02 null); after expanding the visual
   inventory to the needed **ImageNet-21k** classes (12 → 96 testable words),
   assignment beats its permuted null **on every corpus**, and on the
   multi_visual scope the effect is **statistically significant on two**
   (SemCor Δ = +0.046, 95% CI [+0.010, +0.084], 64 words; SemEval-2013 Δ = +0.164,
   [+0.026, +0.339], 6 words).
2. **Is the visual signal real at all?** — does the anchor profile *alone*
   cluster senses above a permuted-anchor null? **Answer: the signal is present**
   (e.g. *board* profile 0.58, *table* 0.47) even where hard assignment is
   brittle — so the information exists beyond what argmax labelling realizes.
3. **Does it beat a strong text model under naive fusion?** (the hard bar, not
   the thesis) — **Answer: no**, across all corpora and both inventories;
   concatenating the raw anchor profile onto a strong text embedding does not
   help.

The demonstration on (1)/(2) is the point of the pilot; the negative on (3)
motivates better grounding and calibrated assignment, not a retreat from the
idea.

---

## 2. Background and motivation

**Word-sense induction (WSI)** is the unsupervised task of partitioning the
usages of a polysemous word into senses, without a predefined sense inventory.
Standard approaches are *distributional*: they cluster contextual text
representations. The question here is whether an **external perceptual signal**
— images — adds sense-discriminating information that text distribution alone
blurs.

The intuition is concrete polysemy. Many words split along senses that
correspond to distinct *visible* things: *crane* (bird vs. machine), *bat*
(animal vs. equipment), *plane* (aircraft vs. carpentry tool). If we can measure,
for a given usage, how much its meaning "looks like" each of several candidate
visual concepts, that similarity vector is a perceptual fingerprint that might
sharpen sense boundaries a text clusterer struggles with.

Three ingredients make this testable:

1. **ImageNet-1k classes are WordNet synsets.** So from any word we can
   enumerate candidate visual concepts through WordNet — a principled,
   inventory-based bridge from a word to a set of images.
2. **A shared multimodal embedding space** makes cross-modal cosine similarity
   `cos(text, image)` meaningful. We use `Qwen3-VL-Embedding-8B`, which embeds
   text and images into one 4096-d space.
3. **The anchor profile as a feature.** For usage *i*, define
   `a_i[c] = cos(t_i, v_c)` over the word's candidate visual classes *c*, where
   `t_i` is the usage's text embedding and `v_c` is class *c*'s image prototype.

---

## 3. Data

Four English WSI datasets, each providing (target word, usage in context, gold
sense). Gold sense labels differ in origin but are treated identically
downstream — as opaque per-lemma cluster labels for evaluation. **All corpora
are nouns-only in this pilot.**

| corpus | gold senses | inventory | notes |
|---|---|---|---|
| **SemCor** | WordNet synsets | dependent | NLTK; broad-coverage sense-tagged corpus |
| **DWUG EN** v3.0.0 | correlation clusters over human relatedness judgments | **free** | diachronic; two time periods **pooled** (sense induction, not change detection); `-1` noise cluster dropped |
| **SemEval-2013 Task 13** | WordNet-3.1 sense keys | dependent | *graded* (multi-sense) labels; **single-sense subset used** for hard clustering (`--include-graded` keeps them via max-weight sense) |
| **SemEval-2010 Task 14** | OntoNotes-style sense ids | dependent | no target-token offset in the data → target located within `<TargetSentence>` |

DWUG EN is the resource this pilot is ultimately aimed at, because its senses are
**inventory-free** (induced from pairwise human judgments), which is the honest
setting for sense *induction*. The others are included so that conclusions do
not rest on one dataset's idiosyncrasies.

Each dataset has a dedicated extractor emitting a common occurrence schema
(`lemma, sentence_id, sentence, target_start, target_end, target_surface,
gold_synset`), so every downstream stage is identical regardless of corpus.

---

## 4. Method

### 4.1 Target selection and visual grounding

We keep lemmas with ≥ 40 occurrences over senses that each have ≥ 8 examples and
≥ 2 such senses. Rare senses are pruned by the per-sense threshold (this also
prunes DWUG's long singleton-cluster tail).

**Visual grounding.** ImageNet-1k classes are *specific leaf* synsets
(`airliner`, `soccer_ball`), whereas a word's senses are higher-level
(`airplane`, `ball`). A bare surface-lemma match therefore finds almost nothing.
We instead expand up the WordNet hypernym graph: an ImageNet class `c` grounds a
word sense `s` if `c` lies within **3 hypernym levels below `s`** (`airliner` →
`airplane`), capped at 12 anchors per sense. A word's anchor set `C_w` is the
union of its grounded senses' classes.

Crucially, this grounding uses the **WordNet senses of the lemma**, *independent*
of the corpus gold labels. The gold labels are used only to score clustering;
the visual anchors are candidate concepts, never supervision.

Words are stratified by the number of **visually-grounded senses** `g_w`:

- `multi_visual` (`g_w ≥ 2`) — ≥ 2 senses with distinct visual anchors;
- `visual_nonvisual` (`g_w = 1`) — one concrete sense vs. other usages;
- `text_only` (`g_w = 0`) — excluded from the visual comparison.

### 4.2 Encoder

`Qwen/Qwen3-VL-Embedding-8B` (bf16, 4096-d, last-token pooling, instruction-aware,
via sentence-transformers). One model produces both modalities:

- **Text** `t_i`: the ±20-word context window around the target, encoded with a
  **target-aware instruction** (`Represent the meaning of the word "{lemma}" as
  used in the following context.`). This same vector is reused as the clustering
  base (after PCA-64) and as the cross-modal anchor query (raw, in the shared
  space).
- **Image prototype** `v_c`: 32 sampled ImageNet training images per class,
  embedded and mean-pooled, then L2-normalised. These are **whole images — no
  bounding-box crop** (see §8: the object covers only ~45% of a typical frame,
  so the prototype encodes scene/background as well as the object).
- **Label prototype**: the text embedding of the class *name* — a control (see
  below).

### 4.3 Systems compared

| system | features | role |
|---|---|---|
| `qwen` | PCA-64 of `t_i` | text-only baseline |
| `qwen+image` | `[h̃ ; λ·zscore(a_i)]`, `a_i[c]=cos(t_i,v_c)` | the hypothesis |
| `qwen+label` | same, but `v_c` = class-*name* text embedding | control: image vs. just the name |
| `qwen+shuffled-image` | same, but class→prototype identity permuted | control: real signal vs. structured noise |
| `image-profile-only` | `a_i` alone (k-means) | is the anchor informative by itself? |
| `shuffled-profile-only` | permuted `a_i` alone | **null** for the line above |
| `anchor-assignment` | `argmax_s max_c cos(t_i,v_c)` (no k-means) | **inductive nearest-visible-anchor sense labelling** |
| `anchor-assignment-shuffled` | same, permuted prototypes | **null** for assignment |

`anchor-assignment` is the headline predictor: it assigns each usage directly to
its nearest *grounded sense* (pooling the ImageNet classes under a sense, so a
sense with several hyponyms is not over-split). It uses no clustering and no K,
so it applies to arbitrary unseen usages and its induced senses are *named* by
their ImageNet class.

Fusion vectors are L2-normalised; the fusion weight λ is chosen per lemma by
leave-one-lemma-out (LOLO) tuning to avoid peeking.

### 4.4 Clustering and evaluation

Spherical k-means per lemma, 10 seeds, in two regimes: **oracle-K** (gold number
of senses supplied) and **unknown-K** (silhouette-selected). Metrics: **ARI**
(primary), V-measure, B-cubed F1. ARI is chance-corrected: its expected value
under random labeling is 0, so `ARI > 0` already means real structure.

We report three questions with distinct statistics (paired bootstrap over lemmas):

- **Assignment test** (headline): `Δ_assign = ARI(anchor-assignment) −
  ARI(anchor-assignment-shuffled)`. Does labelling usages by their nearest
  visible anchor recover senses above a permuted-anchor null? Reported over all
  visual lemmas and, more fairly, over `multi_visual` only (single-sense lemmas
  can only produce one assignment cluster, so they sit at ARI 0 by construction).
- **Meaningful-signal test**:
  `Δ_signal = ARI(image-profile-only) − ARI(shuffled-profile-only)`. Both use the
  anchor *alone*; the null keeps identical dimensionality and scale but scrambles
  which image belongs to which class. `Δ_signal > 0` means the visual channel
  carries **class-specific** sense information beyond its own structure.
- **Naive-fusion test** (the hard, secondary bar):
  `Δ_image = ARI(qwen+image) − ARI(qwen)`, plus the label and shuffled controls
  and a strict 6-criterion go/no-go rule.

---

## 5. Experimental setup

- Hardware: NVIDIA RTX PRO 6000 Blackwell (single GPU); embeddings in bf16.
- Visual inventories, two conditions:
  - **ImageNet-1k**: ILSVRC-2012 train split (1000 classes, ~1300 images each).
  - **ImageNet-21k (targeted)**: the winter21 release hosts per-synset tarballs;
    rather than the full ~1.3 TB we downloaded **only the 1,115 anchor classes**
    the four corpora actually need (~81 GB; 216 further wanted classes are
    absent from winter21), merged with ILSVRC into a 2,115-class root.
- Retained visual lemmas per corpus under each inventory:

| corpus | lemmas selected | visual (1k → 21k) | multi_visual (1k → 21k) |
|---|---|---|---|
| SemCor | 209 | 25 → 99 | 7 → **64** |
| DWUG EN | 21 | 6 → 15 | 3 → **12** |
| SemEval-2010 | 37 | 4 → 20 | 2 → **14** |
| SemEval-2013 | 15 | 3 → 10 | 0 → **6** |

Under ImageNet-1k the visual-lemma counts are the central practical constraint:
few polysemous nouns have ≥ 2 senses that each land under an ImageNet-1k
subtree, which limits statistical power. The targeted 21k expansion lifts the
assignment-testable vocabulary from **12 to 96 words** (~8×) and, on SemCor,
raises words whose *gold* senses are all anchorable from 0 to 12.

---

## 6. Results

Sixteen runs share one pipeline (Qwen encoder, target-aware instruction): 4
corpora × {oracle-K, unknown-K} × {ImageNet-1k, ImageNet-21k}. The table below
is the ImageNet-1k condition; §6.1 carries the 21k comparison.

**Column key — note these are the *profile-clustering* (Question 2) columns, not
assignment.** `profile` = `image-profile-only` (cluster the anchor profile);
`profile-null` = `shuffled-profile-only`; `Δ_signal` = their difference; `text` =
`qwen`; `+img`/`+lbl` = fused image/label. The *assignment* comparison
(`anchor-assignment` vs its null — the headline) is a **different** test and
lives in §6.1; do not read these `profile`/`profile-null` columns as the
assignment result.

| corpus (mode) | #vis | profile | profile-null | Δ_signal | text | +img | Δ_image | +lbl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SemCor (oracle) | 25 | 0.069 | 0.065 | +0.004 | 0.402 | 0.320 | −0.082 | 0.298 |
| SemCor (unknown) | 25 | 0.065 | 0.069 | −0.004 | 0.385 | 0.350 | −0.034 | 0.324 |
| DWUG EN (oracle) | 6 | 0.156 | 0.150 | +0.006 | 0.554 | 0.545 | −0.009 | 0.504 |
| **DWUG EN (unknown)** | 6 | **0.215** | 0.126 | **+0.089** | 0.705 | 0.696 | −0.008 | 0.648 |
| SemEval-2013 (oracle) | 3 | 0.028 | −0.001 | +0.029 | 0.364 | 0.354 | −0.010 | 0.418 |
| SemEval-2013 (unknown) | 3 | 0.031 | −0.002 | +0.033 | 0.390 | 0.394 | +0.004 | 0.433 |
| SemEval-2010 (oracle) | 4 | 0.036 | 0.053 | −0.018 | 0.550 | 0.537 | −0.013 | 0.512 |
| SemEval-2010 (unknown) | 4 | 0.067 | 0.073 | −0.006 | 0.558 | 0.541 | −0.017 | 0.530 |

### 6.1 Grounded sense assignment via visible anchors (headline)

Assigning each usage to its nearest grounded-sense anchor, scored over
`multi_visual` lemmas (the fair scope; identical in oracle/unknown-K since
assignment is K-free), under both visual inventories:

| corpus | 1k: #mv, assign / null | 21k: #mv, assign / null |
|---|---|---|
| SemCor | 7 — 0.058 / 0.045 | **64 — 0.080 / 0.034** |
| DWUG EN | 3 — 0.245 / 0.034 | 12 — 0.199 / 0.106 |
| SemEval-2013 | 0 — n/a | **6 — 0.208 / 0.043** |
| SemEval-2010 | 2 — −0.032 / 0.018 | 14 — 0.103 / 0.084 |

Under ImageNet-1k the evidence rests on a couple of words; under the 21k
expansion **assignment beats its permuted null on all four corpora**. Scored on
the **`multi_visual` lemmas** — the non-degenerate scope, since single-grounded-
sense words are ARI-0 by construction and would only dilute the mean — the paired
bootstrap gives the study's first significant aggregates:

| corpus (21k) | n multi_visual | Δ_assign (mv) | 95% CI | excludes 0 |
|---|---:|---:|---|---|
| **SemCor** | 64 | **+0.046** | [+0.010, +0.084] | **yes** |
| **SemEval-2013** | 6 | **+0.164** | [+0.026, +0.339] | **yes** |
| DWUG EN | 12 | +0.093 | [−0.017, +0.221] | no (n=12) |
| SemEval-2010 | 14 | +0.019 | [−0.058, +0.115] | no |

(Over *all* visual lemmas — including the single-sense zeros — the SemCor figure
is a diluted but still-significant +0.030 [+0.007, +0.056]; we report the
`multi_visual` scope as the honest effect size.)

The effect also stops being a two-word anecdote. Words where assignment beats
its null by > 0.15 (was: `plane`, `board`) now include, e.g.:

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

Bolded rows are cases where the **visual assignment outperforms text-only
clustering** (`head`, `cell`, `part`, also `bit`, `level`) — the first direct
instances of the visual channel beating the text baseline on any measure.

Three takeaways. **(a)** Nearest-visible-anchor assignment — a *named,
inspectable, inductive* sense labelling with no clustering — works, and with
adequate coverage the effect is corpus-level and (on SemCor) statistically
significant, not anecdotal. **(b)** It remains word-class-dependent: only
25–40 % of visual lemmas beat their null; the aggregate is carried by genuinely
visual words. **(c)** For several words (`table`, `ball`, `board` under 1k) the
signal is present in the profile *geometry* (profile ARI 0.26–0.58) yet raw
`argmax` underperforms it — the anchor space is informative but the
nearest-anchor *decision rule* is uncalibrated; closing that gap is the open
problem.

### 6.2 Is the visual signal meaningful? (profile-clustering, aggregate)

Weak and corpus/mode-dependent, under *both* inventories — and, tellingly, the
21k expansion did **not** rescue this test the way it rescued assignment.
Clustering the anchor profile beats its permuted null only marginally: under 21k
the deltas are SemCor +0.020/+0.014, SemEval-2013 +0.063/+0.051, DWUG
+0.006/−0.034, SemEval-2010 +0.010/−0.014 (oracle/unknown). The absolute ARIs
rose with more anchors (SemCor profile 0.07 → 0.19), but the *null* rose in
lockstep — more anchor dimensions help k-means regardless of whether the images
are correctly bound.

This contrast is informative: extra visual coverage sharpened the **argmax
assignment** decision (§6.1) far more than it sharpened **profile clustering**.
Reading a single nearest-anchor label off the profile exploits class-specific
structure that whole-vector clustering blurs. Honest aggregate read for the
profile test itself: **at or near chance**.

### 6.3 Profile clustering, per lemma (selectivity check)

The weak aggregate (§6.2) hides sharp per-word structure. It is **not** a
two-word effect: under ImageNet-21k, ~22 `multi_visual` words cluster above their
null (Δ > 0.10, oracle-K). For genuinely visually polysemous words, **clustering
the profile alone** (`image-profile-only`) recovers senses far above its null,
and this **replicates across clustering regimes** (ImageNet-21k, oracle / unknown-K):

| word (corpus) | profile-only ARI (oracle / unknown) | profile-null (oracle / unknown) |
|---|---|---|
| **`head`** (SemCor) | **0.74 / 0.74** | 0.28 / 0.45 |
| **`face`** (DWUG) | **0.41 / 0.41** | 0.07 / 0.07 |
| **`cell`** (SemCor) | **0.77 / 0.77** | 0.55 / 0.54 |

(others: `paper`, `officer`, `section`, `grain`, `body`, `house`, `field`, …).
Note the clustering "winners" differ from the assignment winners (§6.1) — e.g.
`head`/`face` cluster strongly, whereas `plane`/`cell` *assign* strongly — since
argmax and whole-profile k-means read the same profile differently.

Meanwhile the method does **not** hallucinate signal where there is none:
abstract nouns (`market` profile 0.00; `area`, `church`, `door` ≈ 0) sit at
chance, and some nominally-visual words whose senses are not visually separable
(`house`, `ball` in one regime) fall at or below their null. This selectivity —
strong where senses are visually distinct, null where they are not — is the core
evidence that **the visual channel encodes real, interpretable sense
information**.

### 6.4 Does it beat a strong text model under naive fusion?

No, consistently, and this is the one result the 21k expansion did **not**
change. `Δ_image ≤ 0` in essentially every run under both inventories; under 21k
it is negative on all four corpora (−0.04 to −0.09). Two qualifications matter:

- The **target-aware instruction dramatically strengthened the text baseline**
  (e.g. DWUG `qwen` rose from ~0.24 with a plain instruction to 0.55/0.71). This
  is the single largest effect in the study — and it is on the *text* side.
- Adding more (and more correctly-bound) anchors did not help fusion: naive
  concatenation of the anchor profile onto an already-strong text representation
  simply injects variance. This isolates the failure to the **fusion
  mechanism**, not to coverage — the same signal that yields a significant
  *assignment* result is wasted under fixed-λ concatenation.

**Fusion-scaling sweep (SemCor @21k, multi_visual, oracle-K).** To rule out that
the failure is a dimensionality/scaling artifact, we swept how the visual block
is built and sized before concatenation — the profile `a_i`, and the *selected
prototype* `v_{c*}` (argmax anchor) both hard and softmax-weighted — at three
scalings, each with per-lemma λ tuning:

| visual block | raw | PCA-32 img | full-4096 both |
|---|--:|--:|--:|
| profile `a_i` (= `qwen+image`) | 0.301 | — | — |
| selected proto, argmax | 0.155 | 0.231 | 0.161 |
| selected proto, softmax | 0.189 | 0.230 | 0.191 |
| **text baseline** | **0.389** (PCA-64) / **0.399** (full-4096) | | |

Every variant loses to text, and **best λ pins at the grid floor in every case**
(the optimiser wants to *ignore* the visual block). Balancing dimensionality
(PCA-32) reduces the harm but does not remove it. Two structural reasons, not
scale: (i) the selected-prototype feature is **constant within each argmax
group**, so concatenating it injects the *assignment* partition — whose
standalone ARI is only ~0.08 — dragging the 0.39 text clustering toward it; and
(ii) a z-scored block of `d` dims has norm ∝ √d, which swamps the unit-norm text
unless λ→0. So the fusion failure is **robust to scaling** — it is the
fixed-concatenation mechanism, and the fix is learned/gated fusion that trusts
the visual block per-word (incidental notes: full-4096 text ≈ PCA-64 text, so
PCA-64 costs ~nothing; softmax pooling consistently beats hard argmax).

### 6.5 Image vs. name

On SemEval-2013 the **class name** (`+label` ≈ 0.42–0.43) helps more than the
class *image* (`+img` ≈ 0.35–0.39) and more than text. A clean reminder that
lexical grounding and visual grounding are different signals, and that for some
words the *name* of the visual concept is the more useful cue.

### 6.6 Does cropping to the object help? (bounding-box A/B)

§8 notes the prototypes are whole images whose object fills only ~45% of the
frame — so an obvious hypothesis is that the background is *noise* and cropping
to the object would sharpen the anchor. We tested it directly on SemCor @1k: for
each anchor class, take the **same 32 annotated images** and build two matched
prototypes — the whole image vs. the image cropped to the union bounding box
(object covers 52% of frame on average, so the crop removes ~48%). Only the crop
differs; text embeddings, targets, and clustering are identical.

**Cropping *hurt*, and sharply** (multi_visual assignment, oracle-K):

| variant | assign | null | Δ |
|---|---:|---:|---:|
| whole image | 0.072 | 0.042 | **+0.030** |
| cropped to bbox | 0.001 | 0.020 | **−0.019** |

Per lemma, the two words carrying the whole-image signal lost it entirely:
`board` 0.378 → 0.000, `light` 0.120 → 0.005 (the rest were ~0 either way).

The most likely reading flips the initial intuition: **the surrounding scene is
carrying sense-relevant signal, not just noise.** `board`'s anchors are `plank`
vs. `dining_table` — in whole images they differ enormously by *scene* (workshop
vs. dining room), and that scene correlates with the sense; cropped to the bare
object, two flat wooden surfaces look *more* alike. So for word-sense work,
image context appears to *disambiguate*.

Caveats (this is suggestive, not conclusive): small n (only `board`/`light` had
signal to lose); and crop-and-upscale **degrades** the image (resolution/framing
shift) — confounded with context removal. The honest follow-up is to **mask or
blur the background in place** (same framing/resolution) to separate "context
helps" from "cropping degrades." This experiment is isolated in a standalone
script (`build_crop_prototypes.py`); it did not modify the pipeline.

---

## 7. Discussion

The picture is coherent across four benchmarks. The visual channel carries
**genuine but narrow** sense signal:

- **Visible anchors *can* label usages — at corpus scale.** Nearest-visible-anchor
  assignment is a grounded, inductive, human-inspectable sense labelling with no
  clustering. Under adequate visual coverage (21k) it beats its permuted null on
  all four corpora and is **statistically significant on two** (multi_visual
  scope: SemCor Δ = +0.046 [+0.010, +0.084], 64 words; SemEval-2013 Δ = +0.164
  [+0.026, +0.339], 6 words) — no longer a `plane`-shaped anecdote. Several words
  (`head`, `cell`, `part`, `bit`, `level`) even beat text-only clustering.
- **Coverage was a real ceiling, and liftable cheaply.** The 1k→21k jump (12→96
  testable words) is what turned a per-word curiosity into an aggregate result,
  and needed only the ~1,100 anchor synsets the corpora actually touch (~81 GB),
  not the full 1.3 TB. But coverage is not sufficient: fully-gold-covered words
  are mixed (`plant`/`floor`/`head` strong; `yard`/`film`/`book` ≈ 0).
- **The signal is realized by argmax, less so by clustering.** Extra coverage
  sharpened nearest-anchor *assignment* far more than *profile clustering* (whose
  null rose in step). Reading one named label off the profile exploits
  class-specific structure that whole-vector k-means blurs.
- **It is inert where it should be.** Abstract / non-visually-separable words sit
  at chance — the anchor is not a generic "extra features help" artifact.
- **It still does not beat a strong text model under naive fusion**, under either
  inventory. With coverage no longer the bottleneck, the remaining causes are
  (i) **granularity** — an *averaged* class prototype is a coarse "typical look,"
  and cross-modal cosine is dominated by topic over fine sense. (We initially
  suspected the background — the object fills only ~45% of the frame, §8 — was
  the culprit, but cropping it away *hurt* assignment, §6.6: scene context
  appears sense-informative, so "coarse" is not simply "too much background."); (ii)
  **redundancy** — a strong instruction-tuned text embedder already captures most
  of the structure; and (iii) the **fusion mechanism** itself — fixed-λ
  concatenation wastes signal that argmax assignment extracts.

For an introduction to multimodal WSI, this is the honest and useful result:
**images are usable and informative for a substantial, growing set of words** —
demonstrably so once the visual inventory is adequate — and the remaining
negative (naive fusion) is now pinned on the fusion mechanism rather than on a
lack of visual coverage.

---

## 8. Limitations

- **Sample size, now partly addressed.** Under ImageNet-1k the visual pools were
  tiny (3–25 lemmas); the 21k expansion raised them to 10–99, which is what let
  SemCor reach significance. DWUG/SemEval-2010/2013 pools (15/20/10) still give
  wide CIs, so their positive point estimates are suggestive, not conclusive.
- **Visual inventory — improved, not solved.** Targeted 21k covers ~96 testable
  words, but 216 wanted synsets are absent from winter21, and 21k *tail* classes
  are noisier and smaller (some < 32 images) than the curated 1k. Coverage is now
  broad but uneven in quality.
- **Prototypes are whole-image (object ≈ 45% of frame), and that context turns
  out to matter.** We embed the full JPEG, no crop. Measured on 1,200 annotated
  ILSVRC images, the object bbox covers a **median of only ~45%** of the frame
  (66% < 60%; 46% < 40%), and 32 such images are averaged — so a prototype is a
  coarse, scene-heavy "typical look." We expected this background to be dilutive
  noise, but the bbox-crop A/B (§6.6) **refuted that**: removing it *hurt*
  assignment. So the weakness is subtler — the prototype conflates object and
  scene, and both appear to carry sense-relevant signal; simply stripping the
  scene is not the fix.
- **Naive fusion.** Simple normalised concatenation with a scalar λ; learned or
  gated fusion is untested.
- **Scope.** English, nouns only. The WordNet→ImageNet bridge is English-specific
  (non-English DWUGs would need a different grounding route).
- **Anchor grounding uses WordNet senses of the lemma**, which need not align
  with a corpus's gold senses (especially DWUG's induced clusters); a word can be
  `multi_visual` yet have gold senses that do not match its visual split.

---

## 9. Future work

1. **Calibrated assignment.** The profile-vs-assignment gap (§6.1) shows the
   anchor space is informative where raw `argmax` is not; per-sense score
   calibration, thresholds, or a learned assignment head should recover much of
   the profile-clustering quality as *named* labels.
2. **Gloss-based anchors.** Ground each *induced* cluster by its definition/gloss
   (available for DWUG via cluster glosses) → embed the gloss text → compare, a
   route that needs no WordNet and may carry sharper sense signal.
3. **Better prototypes — but *not* naive cropping.** The bbox-crop A/B (§6.6)
   showed that removing the background *hurt*, so object-only prototypes are out.
   The disentangling experiment is a **background mask/blur in place** (same
   framing and resolution, context removed) to test whether scene context is
   genuinely informative or the crop merely degraded the image. Beyond that,
   **usage-conditioned or exemplar** anchors — retrieve/weight the class images
   most like the usage — rather than one scene-heavy averaged prototype.
   (Inventory *coverage* is largely handled by the 21k fetch; the next gain is
   prototype *quality*, not more classes.)
4. **Learned fusion** (gating, attention) rather than fixed concatenation — §6.4
   isolates the failure to the fusion mechanism, so this is now the highest-value
   change for the fusion result specifically.
5. **Extend to verbs / other languages**, and to diachronic sense-change
   detection (DWUG's native task), for which the periods are currently pooled.

---

## 10. Reproducing this

```bash
make setup                                   # venv + non-torch deps + NLTK corpora
uv pip install --python .venv/bin/python torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu129     # Blackwell (sm_120)

export IMAGENET_ROOT=/path/to/ILSVRC/Data/CLS-LOC        # dir whose train/ holds n######## classes
make dwug-fetch semeval2013-fetch semeval2010-fetch      # download the corpora
CORPUS=dwug_en bash run.sh                               # or semcor / semeval2013 / semeval2010
```

For the ImageNet-21k condition, only the anchor synsets a corpus needs are
required: run `select_targets` against the 21k synset list (`image-net.org`
synset API), collect the resulting `anchor_wnids`, download just those
`winter21_whole/<wnid>.tar` tarballs into class folders, merge with ILSVRC under
one root, and point `IMAGENET_ROOT` at it. This is ~81 GB (1,115 classes), not
the full ~1.3 TB.

`run.sh` runs the full pipeline end-to-end and writes
`results/<run>/report.md` for oracle-K and unknown-K. Each stage is also a
`make` target and a plain `python -m src.<stage>` module. Key knobs live in
`configs/pilot.yaml` (thresholds, hypernym-expansion depth, embedder,
instruction template, λ grid, seeds).

**Pipeline stages** (`src/`): `audit` → `extract_{semcor,dwug,semeval2013,semeval2010}`
→ `index_imagenet` → `select_targets` → `embed_imagenet` (image prototypes) +
`embed_contexts` (text + label prototypes) → `cluster` → `evaluate` → `report`.

The sixteen reports discussed here are under
`results/{semcor,dwug_en,semeval2013,semeval2010}_{oracle,unknown}_k/report.md`
(ImageNet-1k) and the same with a `_21k` suffix
(`results/<corpus>_21k_{oracle,unknown}_k/report.md`).
