# Hybrid text/image gate and the per-word inventory

Two CPU-only re-analyses of artifacts the pipeline already produced, over the
`multi_visual` words of the four `*_21k_oracle_k` runs. Full discussion in
REPORT.md §11.1 (gate) and §9 (inventory).

Scripts: `hybrid_gate.py` (reads `results/<corpus>_21k_oracle_k/
{assignments.parquet,per_lemma.csv}`), `inventory.py` (reads `per_lemma.csv`).

## 1. Gating on text uncertainty — fails, instructively (§11.1)

The natural hybrid: keep text clustering, fall back to image assignment on the
words text can't handle, detected gold-free by cross-seed instability of the
text partition. Both findings are negative:

1. **The ceiling is tiny.** Even an oracle that picks the better system per
   word gains only ~+0.019 ARI pooled; realistic uncertainty gates recover
   ~+0.004 ("always use image" is catastrophic: SemCor 0.389 → 0.080).
2. **The premise is false.** The words image rescues are among text's most
   *stable* words — text is not unsure about them, it is confidently wrong,
   partitioning cleanly by topic/register instead of sense. Any text-internal
   confidence signal rewards exactly those well-separated wrong clusters. The
   correlations corr(uncertainty, image−text) are weak and flip sign across
   corpora (+0.24 SemCor, −0.61 SemEval-2013).

The gates that do work are image-side and text–image signals — see
`experiments/cluster_label/` (§11.2) and `experiments/agreement_gate/`
(§11.3–11.4).

## 2. Per-word inventory, conservative thresholds (§9)

`inventory.py` classifies each of the 96 assignment-testable word–corpus pairs
by Δ = assignment − shuffled-null ARI:

| outcome | Δ | count | share |
|---|---|--:|--:|
| clearly works | ≥ +0.10 | 22 | 23 % |
| marginal | +0.02 to +0.10 | 14 | 15 % |
| no signal | < +0.02 | 60 | 62 % |

The works-rate is broadly similar across corpora (SemCor 13/64, DWUG 4/12,
SemEval-2013 3/6, SemEval-2010 2/14); the full works/marginal tables are in
REPORT.md §9.
