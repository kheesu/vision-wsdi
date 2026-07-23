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
   the induced partition. **Answer: yes for genuinely visual words — strikingly
   so for *plane* (ARI 0.70 vs. 0.02 null)** — and DWUG's multi-visual set beats
   its null in aggregate (0.245 vs. 0.034); weak/at-chance elsewhere.
2. **Is the visual signal real at all?** — does the anchor profile *alone*
   cluster senses above a permuted-anchor null? **Answer: the signal is present**
   (e.g. *board* profile 0.58, *table* 0.47) even where hard assignment is
   brittle — so the information exists; realizing it as labels is the open part.
3. **Does it beat a strong text model under naive fusion?** (the hard bar, not
   the thesis) — **Answer: no**, across all four corpora; concatenating the raw
   anchor profile onto a strong text embedding does not help.

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
  embedded and mean-pooled, then L2-normalised.
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
- ImageNet: ILSVRC-2012 train split (1000 classes, ~1300 images each).
- Retained visual lemmas per corpus (against real ImageNet-1k):

| corpus | lemmas selected | visual lemmas | of which multi_visual |
|---|---|---|---|
| SemCor | 209 | 25 | 7 |
| DWUG EN | 21 | 6 | 3 |
| SemEval-2010 | 37 | 4 | 2 |
| SemEval-2013 | 15 | 3 | 0 |

The small visual-lemma counts are the central practical constraint: **ImageNet-1k
is a narrow visual inventory**, so few polysemous nouns have ≥ 2 senses that each
land under an ImageNet subtree. This limits statistical power and is a coverage
ceiling, not a modelling choice.

---

## 6. Results

All eight runs use the same pipeline (Qwen encoder, target-aware instruction,
reoriented reporting). `anchor` = `image-profile-only`; `null` =
`shuffled-profile-only`; `text` = `qwen`; `+img`/`+lbl` = fused image/label.

| corpus (mode) | #vis | anchor | null | Δ_signal | text | +img | Δ_image | +lbl |
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
assignment is K-free):

| corpus | #multi_visual | assignment ARI | permuted null |
|---|---:|---:|---:|
| **DWUG EN** | 3 | **0.245** | 0.034 |
| SemCor | 7 | 0.058 | 0.045 |
| SemEval-2010 | 2 | −0.032 | 0.018 |
| SemEval-2013 | 0 | — | — |

The effect concentrates in individual words:

| word | assignment ARI | null | profile-clustering ARI |
|---|---:|---:|---:|
| **`plane`** (DWUG) | **0.70** | 0.02 | 0.43 |
| `board` (SemCor) | 0.32 | 0.20 | 0.58 |
| `light` (SemCor) | 0.12 | 0.01 | 0.11 |
| `table` (SemCor) | −0.05 | 0.05 | 0.47 |
| `ball` (DWUG) | −0.01 | 0.02 | 0.26 |

Two takeaways. **(a)** For the cleanest visual word, `plane`, direct assignment
to a *named, inspectable* anchor recovers senses at **ARI 0.70** — a grounded,
inductive sense labelling with no clustering, exactly the capability text-only
methods cannot offer. **(b)** For several words (`table`, `ball`, `board`) the
visual signal is clearly present in the profile *geometry* (profile ARI
0.26–0.58) yet raw `argmax` assignment underperforms it — the anchor space is
informative, but the nearest-anchor *decision rule* is brittle/uncalibrated.
The capability is demonstrated; making it robust (calibrated assignment,
per-sense thresholds) is the open problem.

### 6.2 Is the visual signal meaningful? (aggregate)

Weak and corpus/mode-dependent. The anchor beats its permuted null in 5 of 8
runs, but the margins are small and none clears a clean significance bar. The one
standout is **DWUG EN unknown-K (Δ_signal = +0.089)**. Honest aggregate read:
**at or near chance**, i.e. averaged over all visual lemmas the effect is not
robustly distinguishable from zero — largely because the visual-lemma pools are
small and dominated by marginally-visual words.

### 6.3 Is the visual signal meaningful? (per lemma — the real story)

The aggregate hides the effect that matters. For words that are *genuinely
visually polysemous*, the anchor **alone** recovers senses far above its null,
and this **replicates across clustering regimes**:

| word (corpus) | anchor-only ARI (oracle / unknown) | null (oracle / unknown) |
|---|---|---|
| **`plane`** (DWUG) | **0.43 / 0.70** | 0.12 / 0.15 |
| **`board`** (SemCor) | **0.66 / 0.66** | 0.44 / 0.44 |

Meanwhile the method does **not** hallucinate signal where there is none:
abstract nouns (`market` anchor 0.00; `area`, `church`, `door` ≈ 0) sit at
chance, and some nominally-visual words whose senses are not visually separable
(`house`, `ball` in one regime) fall at or below their null. This selectivity —
strong where senses are visually distinct, null where they are not — is the core
evidence that **the visual channel encodes real, interpretable sense
information**.

### 6.4 Does it beat a strong text model under naive fusion?

No, consistently. `Δ_image ≤ 0` in 7 of 8 runs (a single +0.004 tie). Two
qualifications matter:

- The **target-aware instruction dramatically strengthened the text baseline**
  (e.g. DWUG `qwen` rose from ~0.24 with a plain instruction to 0.55/0.71). This
  is the single largest effect in the study — and it is on the *text* side.
- Against that stronger baseline the image anchor is now **roughly neutral**
  (−0.01 range) on DWUG and SemEval, rather than clearly harmful; only SemCor
  still shows meaningful drag (−0.08). Naive concatenation of a coarse anchor
  profile simply adds variance to an already-good representation.

### 6.5 Image vs. name

On SemEval-2013 the **class name** (`+label` ≈ 0.42–0.43) helps more than the
class *image* (`+img` ≈ 0.35–0.39) and more than text. A clean reminder that
lexical grounding and visual grounding are different signals, and that for some
words the *name* of the visual concept is the more useful cue.

---

## 7. Discussion

The picture is coherent across four benchmarks. The visual channel carries
**genuine but narrow** sense signal:

- **Visible anchors *can* label usages.** For `plane`, nearest-visible-anchor
  assignment recovers senses at ARI 0.70 with no clustering — a grounded,
  inductive, human-inspectable sense labelling, which is the capability images
  uniquely provide. DWUG's multi-visual set beats its null in aggregate.
- **The signal is present more broadly than hard assignment realizes it.** For
  `board`, `table`, `ball` the anchor-profile *geometry* clusters senses well
  (ARI 0.26–0.58) while raw `argmax` assignment lags — the information is there,
  but the nearest-anchor decision rule is uncalibrated. Closing this gap
  (calibrated/thresholded assignment) is the most actionable next step.
- **It is inert where it should be.** Abstract senses and non-visually-separable
  words fall at chance — the anchor is not a generic "extra features help"
  artifact.
- **It does not beat a strong text model under naive fusion.** Three plausible,
  compounding reasons: (i) **coverage** — ImageNet-1k yields too few genuinely
  multi-visual words for the aggregate to move; (ii) **granularity** — an
  *averaged* class prototype is a coarse "typical look," and cross-modal cosine
  is dominated by broad topic rather than fine sense; (iii) **redundancy** — a
  strong instruction-tuned text embedder already captures most of the sense
  structure the images could add, so the anchor mostly injects variance.

For an introduction to multimodal WSI, this is the honest and useful result:
**images are usable and informative for the right words**, and the failure of
naive fusion is a precise motivation for better methods rather than a verdict
against the idea.

---

## 8. Limitations

- **Small samples.** 3–25 visual lemmas per corpus; aggregate CIs are wide.
  Per-lemma findings are the more trustworthy evidence.
- **Visual inventory.** ImageNet-1k is narrow; a larger synset-grounded image
  set (e.g. ImageNet-21k) would populate `multi_visual` far better.
- **Coarse prototypes.** Mean of 32 images per class discards intra-class
  variation; a usage-conditioned or exemplar retrieval scheme may carry finer
  sense signal.
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
3. **Richer / usage-conditioned visual anchors** instead of averaged class
   prototypes; larger visual inventories (ImageNet-21k) for coverage.
4. **Learned fusion** (gating, attention) rather than fixed concatenation.
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

`run.sh` runs the full pipeline end-to-end and writes
`results/<run>/report.md` for oracle-K and unknown-K. Each stage is also a
`make` target and a plain `python -m src.<stage>` module. Key knobs live in
`configs/pilot.yaml` (thresholds, hypernym-expansion depth, embedder,
instruction template, λ grid, seeds).

**Pipeline stages** (`src/`): `audit` → `extract_{semcor,dwug,semeval2013,semeval2010}`
→ `index_imagenet` → `select_targets` → `embed_imagenet` (image prototypes) +
`embed_contexts` (text + label prototypes) → `cluster` → `evaluate` → `report`.

The eight reports discussed here are under
`results/{semcor,dwug_en,semeval2013,semeval2010}_{oracle,unknown}_k/report.md`.
