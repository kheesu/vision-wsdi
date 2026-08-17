# vision-lsi — visual anchors for lexical sense induction (pilot)

A self-contained sub-experiment of the MMEG repo. It tests **one** narrow claim:

> Do ImageNet-derived **visual anchor** features improve unsupervised clustering
> of contextual word usages beyond text-only clustering and beyond an equivalent
> **text-label** anchor?

This is a *proof of signal*, not yet inventory-free lexical semantic induction:
the visual anchors are looked up through WordNet because ImageNet classes are
WordNet synsets (SemCor senses are directly compatible with ImageNet WNIDs).
Optimal transport and full distributional alignment are still out of scope until
this result is positive.

**The full write-up — method, results, audits — is [REPORT.md](REPORT.md).**
The headline: label each usage by its nearest *visible* anchor — no clustering,
no K — and the result beats a permuted-photo control on all four corpora, with
sense labels that come out named and inspectable:

![One word, several pictures: usages of "plane" and "cell" grouped by their nearest ImageNet anchor image, with no sense labels and no text supervision](figure_word_senses.png)

## Corpora

The gold sense labels come from one of four interchangeable, English corpora,
selected by `CORPUS=` on `run.sh`/`make` (each has a `make <name>-fetch` target
and honours a `*_ROOT` env var):

- **`semcor`** (default) — WordNet-annotated senses. Inventory-*dependent*: the
  gold senses are a predefined synset inventory.
- **`dwug_en`** — [DWUG EN](https://www.ims.uni-stuttgart.de/data/wugs)
  (Diachronic Word Usage Graphs). Gold senses are the inventory-*free* clusters
  induced by correlation clustering over human semantic-proximity judgments —
  the direction this pilot is ultimately headed. The two time periods are
  **pooled** (sense induction, not change detection); DWUG's `-1` noise cluster
  is dropped and its singleton clusters are pruned by
  `data.min_occurrences_per_sense`, exactly as rare WordNet senses are pruned.
- **`semeval2013`** — [SemEval-2013 Task 13](https://doi.org/10.5281/zenodo.5638384)
  (WSI for graded senses). Gold = WordNet-3.1 sense keys. Instances are *graded*
  (multiple weighted senses); for hard clustering the **single-sense** subset is
  used by default (graded instances dropped, `--include-graded` to keep them via
  their max-weight sense).
- **`semeval2010`** — [SemEval-2010 Task 14](https://doi.org/10.5281/zenodo.5638549)
  (WSI & Disambiguation). Gold = one OntoNotes-style sense id per instance. The
  target has no offset annotation, so it is located within `<TargetSentence>`
  (whole-word match, falling back to the whole sentence).

All four are English, so the WordNet→ImageNet visual-anchor bridge applies
uniformly. Nouns only in the pilot. Every extractor emits the same occurrence
schema, so downstream stages are identical regardless of corpus.

## What it does

1. **Corpus → occurrences** — one row per sense-annotated noun occurrence
   (lemma, sentence, target span, gold sense), from SemCor or DWUG EN (see
   **Corpora** below).
2. **Target selection** — keep lemmas with enough occurrences and ≥2 senses.
   A word sense is *visually grounded* when an ImageNet leaf class sits within
   `anchor_max_hypernym_dist` WordNet hypernym levels below it (ImageNet-1k
   classes are specific, so a bare-lemma match finds almost nothing; the
   hypernym expansion is what bridges `airliner`→`airplane`). Subsets are keyed
   on the number of grounded senses g_w: `multi_visual` (≥2), `visual_nonvisual`
   (=1), `text_only` (=0). The anchor set is the union of the grounded senses'
   classes, capped at `anchor_max_per_sense` each.
3. **Encoder** — one multimodal model, `Qwen/Qwen3-VL-Embedding-8B`, embeds both
   text and images into a shared 4096-d space. It produces the word-in-context
   text vector t_i (a local window, default instruction) — reused as the
   clustering base via global PCA-64 — and the ImageNet image vectors averaged
   and normalised into class prototypes v_c.
4. **Systems** — `qwen` (text baseline), `qwen+image`, `qwen+label`,
   `qwen+shuffled-image`, `image-profile-only`. The visual anchor profile is
   `a_i[c] = cos(t_i, v_c)` (raw, un-PCA'd t_i, in the shared space); fusion is
   `normalize([h~ ; λ·zscore(a_i)])`.
5. **Clustering** — spherical k-means per lemma, oracle-K and unknown-K
   (silhouette-selected), 10 seeds.
6. **Evaluation** — ARI / V-measure / B-cubed, macro + per-lemma + per-subset,
   leave-one-lemma-out λ tuning, paired bootstrap CIs, and a go/no-go report.

## Setup

```bash
cd vision-lsi
make setup                      # uv venv + non-torch deps + nltk corpora
# CUDA torch build matching the box. RTX PRO 6000 / Blackwell is sm_120 and
# needs cu129 (or cu130) wheels; older CUDA builds will not run on it.
uv pip install --python .venv/bin/python torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu129
```

`make setup` downloads the `semcor` and `wordnet` NLTK corpora. The embedder
`Qwen/Qwen3-VL-Embedding-8B` (~16 GB, bf16) is downloaded from HuggingFace on the
first embedding run and needs a GPU with ≥20 GB VRAM.

## Run

```bash
export IMAGENET_ROOT=/path/to/imagenet   # dir of n######## class folders, or its parent of train/
bash run.sh
```

To run on another corpus, fetch it once and set `CORPUS`:

```bash
make dwug-fetch        && CORPUS=dwug_en     bash run.sh
make semeval2013-fetch && CORPUS=semeval2013 bash run.sh
make semeval2010-fetch && CORPUS=semeval2010 bash run.sh
```

Each corpus honours a `*_ROOT` env var for its extracted location (defaults
`data/dwug_en`, `data/semeval2013`, `data/semeval2010`).

If `IMAGENET_ROOT` is unset or missing, the pipeline still runs end-to-end: the
image-dependent systems are skipped and the text-only baselines plus the
`qwen+label` control are evaluated. `src/audit.py` records the box state and
whether ImageNet was found.

Individual stages are also `make` targets (`make data`, `make contexts`,
`make cluster`, …) and plain module invocations — see `run.sh`.

## Layout

```
configs/pilot.yaml        all knobs (env ${IMAGENET_ROOT} expanded at load)
src/pilotlib/             config, encoders, wordnet utils, metrics
src/{audit,extract_semcor,extract_dwug,extract_semeval2013,extract_semeval2010}.py
src/{index_imagenet,select_targets}.py
src/{embed_imagenet,embed_contexts,construct_features}.py
src/{cluster,evaluate,report}.py
data/  cache/  results/   generated artifacts (git-ignored)
experiments/              follow-up experiments (each with RESULTS.md) + figure scripts
```

## Figures

`make figures` regenerates the poster figures into `figures/` (git-ignored).
The generators live in `experiments/figures/`; they read the pipeline outputs
in `results/` and `cache/`, and the two photo-tile figures additionally sample
ImageNet-21k — point `IMAGENET_DIR` at its train split. The qualitative figure
above (`figure_word_senses.png`) is committed directly.

## Go/no-go

Proceed to a larger experiment when, on the visual subsets:
`macroARI(qwen+image) − macroARI(qwen) ≥ 0.03`, the paired bootstrap CI excludes
zero, `qwen+image > qwen+label`, a majority of multi-visual lemmas improve, the
shuffled control shows no comparable gain, and no single lemma dominates the
aggregate improvement. `src/report.py` evaluates these and writes the verdict to
`results/<run>/report.md`.

**Outcome** (details in [REPORT.md](REPORT.md)): the fusion criterion was not
met — naive `qwen+image` fusion does not beat the text baseline. The pilot's
positive result sits one level down: nearest-visible-anchor *assignment*
carries real signal on all four corpora (REPORT §6), the effect is strongly
word-dependent (§9), and the reliable words can be identified without gold
labels (§11–12). The constructive endpoint is a grounding-and-naming layer on
top of text clustering, not a fusion feature.
