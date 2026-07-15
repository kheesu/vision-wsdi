# vision-lsi — visual anchors for lexical sense induction (pilot)

A self-contained sub-experiment of the MMEG repo. It tests **one** narrow claim:

> Do ImageNet-derived **visual anchor** features improve unsupervised clustering
> of contextual word usages beyond text-only clustering and beyond an equivalent
> **text-label** anchor?

This is a *proof of signal*, not yet inventory-free lexical semantic induction:
the visual anchors are looked up through WordNet because ImageNet classes are
WordNet synsets (SemCor senses are directly compatible with ImageNet WNIDs).
Diachronic corpora, optimal transport, and full distributional alignment are
explicitly out of scope until this result is positive.

## What it does

1. **SemCor → occurrences** — one row per sense-annotated noun occurrence
   (lemma, sentence, target span, gold synset).
2. **Target selection** — keep lemmas with enough occurrences and ≥2 senses;
   assign each to a subset by its visual-anchor count |C_w|:
   `multi_visual` (≥2 anchors), `visual_nonvisual` (=1), `text_only` (=0).
3. **Encoders** — BERT target-token contexts (last-4-layer mean, L2, global
   PCA-64) and CLIP context-text embeddings; CLIP image prototypes per ImageNet
   class (32 images, averaged, normalised).
4. **Systems** — `bert`, `clip-context`, `bert+image`, `bert+label`,
   `bert+shuffled-image`, `image-profile-only`. The visual anchor profile is
   `a_i[c] = cos(t_i, v_c)`; fusion is `normalize([h~ ; λ·zscore(a_i)])`.
5. **Clustering** — spherical k-means per lemma, oracle-K and unknown-K
   (silhouette-selected), 10 seeds.
6. **Evaluation** — ARI / V-measure / B-cubed, macro + per-lemma + per-subset,
   leave-one-lemma-out λ tuning, paired bootstrap CIs, and a go/no-go report.

## Setup

```bash
cd vision-lsi
make setup                      # uv venv + non-torch deps + nltk corpora
. .venv/bin/activate
uv pip install torch torchvision   # a CUDA build matching the box (Blackwell -> recent wheel)
```

`make setup` downloads the `semcor` and `wordnet` NLTK corpora.

## Run

```bash
export IMAGENET_ROOT=/path/to/imagenet   # dir of n######## class folders, or its parent of train/
bash run.sh
```

If `IMAGENET_ROOT` is unset or missing, the pipeline still runs end-to-end: the
image-dependent systems are skipped and the text-only baselines plus the
`bert+label` control are evaluated. `src/audit.py` records the box state and
whether ImageNet was found.

Individual stages are also `make` targets (`make data`, `make contexts`,
`make cluster`, …) and plain module invocations — see `run.sh`.

## Layout

```
configs/pilot.yaml        all knobs (env ${IMAGENET_ROOT} expanded at load)
src/pilotlib/             config, encoders, wordnet utils, metrics
src/{audit,extract_semcor,index_imagenet,select_targets}.py
src/{embed_imagenet,embed_contexts,construct_features}.py
src/{cluster,evaluate,report}.py
data/  cache/  results/   generated artifacts (git-ignored)
```

## Go/no-go

Proceed to a larger experiment when, on the visual subsets:
`macroARI(bert+image) − macroARI(bert) ≥ 0.03`, the paired bootstrap CI excludes
zero, `bert+image > bert+label`, a majority of multi-visual lemmas improve, the
shuffled control shows no comparable gain, and no single lemma dominates the
aggregate improvement. `src/report.py` evaluates these and writes the verdict to
`results/<run>/report.md`.
