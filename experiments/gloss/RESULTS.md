# Gloss-based sense anchors — DWUG-EN prototype

Anchor each DWUG gold cluster by its **gloss** (definition, from
`ltgoslo/wugs_with_definitions`), embed with Qwen3-VL text, and assign each
usage to `argmax cos(context, gloss)`. Anchors are 1:1 with the gold clusters —
no WordNet, no ImageNet, no alignment gap — so accuracy is well-defined.

## Result (all 21 retained DWUG words, oracle scope)

| metric | value |
|---|---|
| macro ARI | **0.307** |
| random-gloss null (real defs of unrelated senses, 5 draws) | 0.023 |
| Δ vs null | **+0.285** — 17/21 words beat null |
| macro accuracy | **0.646** (chance ≈ 0.409) |
| coverage | **21 words** |

Reference — WordNet→ImageNet *image* anchors: assignment 0.199 over only 6
`multi_visual` words (Δ +0.093 vs its null).

Strong: graft 0.89, ball 0.77, bar 0.69, edge 0.68, plane 0.67, land 0.64.
Weak (fine-grained / near-synonymous clusters): head, heel, rag, twist, face.

## Takeaway
The gloss route works far better and broader than the image route — because the
anchors align 1:1 with the evaluated senses and there is no coverage bottleneck.
This is the **text-gloss variant** (definition-matching WSD, à la Lesk/GlossBERT),
**not multimodal**: it shows the alignment+coverage fix is the dominant lever and
that definitions carry strong sense signal. The multimodal continuation is
gloss→image (retrieve/generate an image per definition).

## Multimodal continuation — gloss → image retrieval

Route each sense **gloss through the image space**: embed the definition, retrieve
its top-k=5 nearest ImageNet class-image prototypes (cross-modal, one Qwen space)
from a broad bank of **2,115 merged-21k classes** (16 imgs/class), average them
into a *visual* anchor, and assign usages by that. Compare three anchors per word
over the 21 DWUG words: gloss-**text** (definition embedding directly),
gloss-**image** (the retrieved visual anchor), and a per-word **blend** (best α).

| anchor | macro ARI | coverage |
|---|--:|--:|
| gloss-text (definition embedding) | **0.308** | 21 words |
| gloss-image (retrieved visual anchor) | 0.176 | 21 words |
| blend (oracle per-word α) | 0.334 | 21 words |

Retrieval is sane (`ball`→ball.n.01; `grain`/corn→millet, kernel, corn;
`plane`/aircraft→elevator, horizontal_stabilizer), but **routing the gloss through
image loses signal**: gloss-image (0.176) trails gloss-text (0.308), and the
oracle blend edges text by only +0.026 — image > text on just **5/21** words
(the visual ones: `land`, `ball`, `plane`, `face`, `head`; `head` is the notable
case where text glosses fail, −0.06, but retrieved images help, +0.07).

**Takeaway.** The image is a *lossy intermediary* for the gloss here — the
definition text already carries the sense signal, and passing it through
image-retrieval only discards some. Consistent with the main report: images
produce meaningful, above-null signal and help precisely on the visually-distinct
senses, but do not beat the text route overall.

## Reproduce
    git clone https://github.com/ltgoslo/wugs_with_definitions <root>
    python -m src.extract_dwug_glosses --gloss-root <root> --output data/dwug_glosses.csv
    .venv/bin/python experiments/gloss/eval_gloss_dwug.py         # text-gloss anchors
    # multimodal continuation (needs the broad image bank first):
    IMAGENET_ROOT=/cldata/ImageNet-merged21k \
      .venv/bin/python experiments/gloss/build_image_bank.py      # -> cache/img_bank.pt (2115 classes)
    .venv/bin/python experiments/gloss/eval_gloss_image_dwug.py   # gloss-text vs gloss-image vs blend
    # all need data/dwug_occurrences.parquet + data/dwug_glosses.csv + GPU
