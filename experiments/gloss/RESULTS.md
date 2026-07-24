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

## Reproduce
    git clone https://github.com/ltgoslo/wugs_with_definitions <root>
    python -m src.extract_dwug_glosses --gloss-root <root> --output data/dwug_glosses.csv
    python experiments/gloss/eval_gloss_dwug.py   # needs data/dwug_occurrences.parquet + GPU
