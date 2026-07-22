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

## Corpora

The gold sense labels come from one of two interchangeable corpora, selected by
`data.corpus` (or `CORPUS=` on `run.sh`/`make`):

- **`semcor`** (default) — WordNet-annotated senses. Inventory-*dependent*: the
  gold senses are a predefined synset inventory.
- **`dwug_en`** — [DWUG EN](https://www.ims.uni-stuttgart.de/data/wugs)
  (Diachronic Word Usage Graphs). Gold senses are the inventory-*free* clusters
  induced by correlation clustering over human semantic-proximity judgments —
  the direction this pilot is ultimately headed. The two time periods are
  **pooled** (sense induction, not change detection); DWUG's `-1` noise cluster
  is dropped and its singleton clusters are pruned by
  `data.min_occurrences_per_sense`, exactly as rare WordNet senses are pruned.
  Only DWUG EN is wired in — the visual-anchor lookup still routes through
  WordNet, so non-English DWUGs would need a different anchor bridge.

Both extractors emit the same occurrence schema, so every downstream stage is
identical regardless of corpus.

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

To run on DWUG EN instead of SemCor, fetch the dataset once and set `CORPUS`:

```bash
make dwug-fetch                 # downloads dwug_en v3.0.0 into data/dwug_en
CORPUS=dwug_en bash run.sh      # or: make data CORPUS=dwug_en, etc.
```

`DWUG_EN_ROOT` overrides the dataset location (default `data/dwug_en`).

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
src/{audit,extract_semcor,extract_dwug,index_imagenet,select_targets}.py
src/{embed_imagenet,embed_contexts,construct_features}.py
src/{cluster,evaluate,report}.py
data/  cache/  results/   generated artifacts (git-ignored)
```

## Go/no-go

Proceed to a larger experiment when, on the visual subsets:
`macroARI(qwen+image) − macroARI(qwen) ≥ 0.03`, the paired bootstrap CI excludes
zero, `qwen+image > qwen+label`, a majority of multi-visual lemmas improve, the
shuffled control shows no comparable gain, and no single lemma dominates the
aggregate improvement. `src/report.py` evaluates these and writes the verdict to
`results/<run>/report.md`.
