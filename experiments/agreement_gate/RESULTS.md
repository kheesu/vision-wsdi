# Text–image agreement gate, §12 audits, and a label-assignment control

Four follow-ups to the groundability line (§12), all CPU-only re-analyses of
artifacts the pipeline already produced. Over the 96 `multi_visual` words of the
four `*_21k_oracle_k` runs unless noted.

Scripts: `agreement_gate.py` (findings 1–3, from `results/<corpus>_21k_oracle_k/
{assignments.parquet,per_lemma.csv}`), `label_assignment.py <corpus>` (finding 4,
from the currently-cached corpus). Outputs: `agreement_gate.csv`,
`label_assignment_semeval2010.csv`.

---

## 1. A stronger gold-free gate: does the image assignment *agree with text*?

§12's behavioral signal asks whether the anchor assignment collapses onto one
sense. A second gold-free signal is available from artifacts we already compute:
**ARI between the text (`qwen`) partition and the anchor-assignment partition**,
averaged over the text seeds — i.e. use the strong text clustering as a
*pseudo-gold* to certify the image channel per word. It predicts the outcome
Δ = assignment ARI − shuffled null much better than collapse does:

| gold-free signal | Spearman(·, Δ) |
|---|--:|
| §12 behavioral (max anchor share) | −0.40 |
| **agreement** ARI(text, assignment) | **+0.53** |

Gating on both (keep words with `max share < 0.80` **and** `agreement > 0.10`)
roughly doubles the §12 gate's precision and effect size at moderate recall cost:

| gate | keep | precision | recall | mean Δ kept | mean Δ rejected |
|---|--:|--:|--:|--:|--:|
| §12 behavioral (`share < 0.80`) | 35/96 | 0.43 | 0.68 | +0.134 | +0.010 |
| **+ agreement (`> 0.10`)** | 17/96 | **0.71** | 0.55 | **+0.260** | +0.011 |

The agreement signal is *deliberately* semi-circular — text ≈ gold on most
words, so agreeing with text proxies agreeing with gold. That is exactly what
makes it a usable gold-free certificate: it answers "can the grounded, named,
inductive labelling be trusted on this word?", which is the question the
naming/interpretability use-case (cluster-then-label, variant (a)) needs.

## 2. The limit: rescues are invisible to *every* gold-free signal

What agreement cannot do is find the **rescue words** — the 13 words where the
image assignment beats text-only clustering (`head`, `cell`, `part`, `bit`,
`center`, …). Spearman(agreement, image − text) = **+0.12**, and the quadrant
decomposition shows why:

| quadrant | n | mean Δ | works (Δ≥0.10) | rescues (img>text) |
|---|--:|--:|--:|--:|
| collapsed (`share ≥ 0.80`) | 61 | +0.01 | 7 | 7 |
| spread + agrees with text | 17 | **+0.26** | 12 | **2** |
| spread + disagrees with text | 18 | +0.01 | 3 | 4 |

The combined gate captures only **2 of 13** rescues: a word where image beats
text is a word where text is *wrong*, so agreement with text is low there by
construction — and raw disagreement doesn't identify them either (the
spread+disagree quadrant sits at Δ +0.01; most disagreement is the image
splitting a wrong axis). This sharpens §11: text's failures are invisible to
text-internal confidence (§11), to the image system's own behavior (§12), *and*
to cross-system disagreement. **Gold-free certification of the visual channel is
possible; gold-free rescue detection is not.** The image channel's defensible
unsupervised role is therefore a *certified grounding/naming layer*, not a
text-fixer.

## 3. Audit: §12's behavioral signal is partly mechanical

A collapsed assignment is a near-one-cluster partition, whose ARI is ≈ 0 **by
construction** — so "collapse predicts Δ ≈ 0" is partly tautological, not
evidence that collapse measures visual separability. Decomposed: of the 61
collapsed words, only 10 have assignment ARI > 0.05 at all; *within* the 35
non-collapsed words the correlation drops to Spearman −0.27 and the works-rate
is only 0.43. The gate remains operationally valid (degenerate output is useless
whatever its cause), but its diagnostic content is thinner than §12's framing —
the agreement signal above carries the actual predictive power.

Also surfaced: the shuffled-assignment **null is inflated on some words**
(`cell`@SemEval-2010 0.43, `house` 0.29, `board` 0.28, `ball` 0.20 — shuffled
prototypes also collapse, and a collapsed partition can align with a skewed
gold). Corpus-wide the skew correlation is weak (+0.08), so aggregates stand,
but §6.7's strongly *negative* Δ rows (`board`, `house`, `ball`) mostly measure
a lucky null, not images actively misleading.

## 4. Label-assignment control: the image matters, and image+name fuses free

The pipeline has a `qwen+label` *fusion* control but no label-prototype
*assignment* — the headline §6.1 never controlled for whether the image content
matters or just the class *name*. On the cached corpus (SemEval-2010 @21k, 14
`multi_visual` lemmas; pipeline null for image assignment ≈ 0.084, text 0.616):

| assignment variant | mean ARI |
|---|--:|
| image prototypes (pipeline system) | 0.107 |
| class-name (label) prototypes | 0.079 |
| image, per-sense z-calibrated | 0.082 |
| label, per-sense z-calibrated | 0.142 |
| **0.5·image + 0.5·label score fusion** | **0.148** |

Three reads:

- **Image > name** on raw argmax — the visual content adds over the WordNet
  class identity, supporting the thesis. (Worth replicating on SemCor, where the
  headline lives; needs one re-embed since the cache is single-corpus.)
- **Per-sense z-calibration (§9 future-work 1) is a trade, not a win**: it
  rescues collapse-prone words (`body` 0.01→0.32, `house` 0.04→0.13) but
  destroys the strong ones (`cell` 0.93→0.24, `officer` 0.37→0.17). Natural
  refinement: apply calibration *only* when the raw assignment collapses —
  §12's max-share statistic is exactly the trigger.
- **Score-level image+name fusion is a near-free upgrade**: beats both channels,
  keeps `cell` at 0.93, and lifts `body` 0.01→**0.62**. Name and image anchors
  are complementary *at the assignment level*, unlike feature-level fusion
  (§6.4), which fails. No matched shuffled null was run for the fused scores.

## Reproduce

    # findings 1–3 (needs results/<corpus>_21k_oracle_k artifacts):
    .venv/bin/python experiments/agreement_gate/agreement_gate.py
    # finding 4 (needs the per-corpus caches; <name> labels the output csv):
    .venv/bin/python experiments/agreement_gate/label_assignment.py semeval2010
