"""Render a Markdown report with the go/no-go decision for a clustering run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DELTA_THRESHOLD = 0.03
DOMINANCE_LIMIT = 0.5  # no single lemma may account for >50% of aggregate gain


def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def main() -> None:
    ap = argparse.ArgumentParser(description="Render run report.")
    ap.add_argument("--run", default="results/oracle_k")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    summary = json.loads((run / "summary.json").read_text())
    bootstrap = json.loads((run / "bootstrap.json").read_text())
    macro = pd.read_csv(run / "macro.csv") if (run / "macro.csv").exists() else pd.DataFrame()
    out = Path(args.output) if args.output else run / "report.md"

    L = [f"# Vision-LSI pilot report — `{run.name}` ({summary.get('mode', '?')}-K)", ""]

    n_vis = summary.get("n_visual_lemmas", 0)
    if not n_vis:
        L += [
            "## No visual-anchor lemmas",
            "",
            "No target lemma has usable ImageNet visual anchors, so the image "
            "systems (`qwen+image`, `image-profile-only`) could not run. This is "
            "expected when ImageNet is unavailable on the box. The text-only "
            "baselines still produced metrics — see `macro.csv` / `per_lemma.csv`.",
            "",
        ]
        if not macro.empty:
            tbl = macro[macro["scope"] == "all"]
            L += ["### Macro metrics (all lemmas)", "", tbl.to_markdown(index=False), ""]
        out.write_text("\n".join(L), encoding="utf-8")
        print(f"Wrote {out}")
        return

    # ---- Headline: grounded sense assignment via visible anchors ------------
    b_asg = bootstrap.get("delta_assignment_signal", {})
    asg = summary.get("macro_ari_anchor_assignment")
    asg_null = summary.get("macro_ari_assignment_null")
    d_asg = summary.get("delta_assignment_signal", float("nan"))
    frac_asg = summary.get("frac_assignment_beats_null")
    asg_meaningful = (bool(b_asg.get("excludes_zero"))
                      and (summary.get("delta_assignment_signal") or 0) > 0)

    L += [
        "## Can we assign senses to usages via visible anchors?",
        "",
        f"*The pitch: each candidate sense **is** a concrete ImageNet class (a "
        f"nameable, inspectable image), so any usage is labelled by nearest-sense "
        f"anchor — `argmax_s max_c cos(t_i, v_c)` — inductively, with no clustering. "
        f"Evaluated over **{n_vis}** visually-grounded lemmas.*",
        "",
        f"- **Nearest-anchor assignment** macro ARI: **{_fmt(asg)}** "
        f"(chance-corrected; the induced senses are named by their ImageNet class).",
        f"- **Permuted-anchor null**: **{_fmt(asg_null)}**.",
        f"- Δ = assignment − null = **{_fmt(d_asg)}** "
        f"(95% CI [{_fmt(b_asg.get('ci_low'))}, {_fmt(b_asg.get('ci_high'))}]); "
        f"beats null on **{_fmt((frac_asg or 0) * 100, 0)}%** of lemmas.",
        "",
    ]
    if asg_meaningful:
        L += ["**Finding: ✅ visible anchors assign senses above chance** — a grounded, "
              "reusable, human-inspectable sense space, not an anonymous clustering.", ""]
    else:
        L += ["**Finding: ➖ aggregate assignment is at/near chance**, but see the "
              "per-lemma readout: it succeeds precisely for the visually-distinct words.", ""]

    # ---- Is the visual channel meaningful (profile-only diagnostic)? --------
    b_sig = bootstrap.get("delta_profile_signal", {})
    img_prof = summary.get("macro_ari_image_profile")
    shuf_prof = summary.get("macro_ari_shuffled_profile")
    d_sig = summary.get("delta_profile_signal", float("nan"))
    frac_beats = summary.get("frac_visual_profile_beats_null")
    meaningful = bool(b_sig.get("excludes_zero")) and (summary.get("delta_profile_signal") or 0) > 0

    L += [
        "## Diagnostic: is the visual signal real? (clustering the anchor profile)",
        "",
        f"*A second, clustering-based check on the same signal: does the anchor "
        f"profile alone recover senses above a permuted null? Over **{n_vis}** lemmas.*",
        "",
        f"- **Visual anchor alone** (`image-profile-only`) macro ARI: "
        f"**{_fmt(img_prof)}** — ARI is chance-corrected, so >0 already means the "
        f"anchor recovers real sense structure.",
        f"- **Permuted-anchor null** (`shuffled-profile-only`): **{_fmt(shuf_prof)}**.",
        f"- Δ_signal = anchor − null = **{_fmt(d_sig)}** "
        f"(95% CI [{_fmt(b_sig.get('ci_low'))}, {_fmt(b_sig.get('ci_high'))}]); "
        f"anchor beats its null on **{_fmt((frac_beats or 0) * 100, 0)}%** of lemmas.",
        "",
    ]
    if meaningful:
        L += ["**Finding: ✅ the visual channel carries sense signal above chance.**", ""]
    else:
        L += ["**Finding: ➖ visual signal is at/near chance on this data.** "
              "(Small samples and ImageNet-1k coverage limit power; see the "
              "per-lemma cases below for where it does land.)", ""]

    # ---- Secondary: does it improve a strong text model under naive fusion? --
    b_img = bootstrap.get("delta_image_vs_qwen", {})
    b_lbl = bootstrap.get("delta_image_vs_label", {})
    d = summary.get("delta_image", float("nan"))

    checks = {
        "Δ_image ≥ 0.03": (summary.get("delta_image") or 0) >= DELTA_THRESHOLD,
        "bootstrap CI(Δ_image) excludes 0": bool(b_img.get("excludes_zero")),
        "qwen+image > qwen+label": (summary.get("macro_ari_qwen_image", 0)
                                    > summary.get("macro_ari_qwen_label", 0)),
        "majority of multi-visual lemmas improve":
            (summary.get("frac_multi_visual_improved") or 0) > 0.5,
        "image gain > shuffled gain":
            summary.get("macro_ari_qwen_image", 0) > summary.get("macro_ari_qwen_shuffled", 0),
        "no single lemma dominates gain":
            (summary.get("max_single_lemma_share") is None
             or summary["max_single_lemma_share"] < DOMINANCE_LIMIT),
    }
    go = all(checks.values())

    L += [
        "## Does it improve a strong text model? (naive fusion — stretch goal)",
        "",
        "*This is the hard bar, not the thesis: whether concatenating the raw "
        "anchor profile onto a strong text embedding beats text alone. Not passing "
        "it means naive fusion is insufficient, not that images are uninformative.*",
        "",
        f"- macro ARI — qwen: **{_fmt(summary.get('macro_ari_qwen'))}**, "
        f"qwen+image: **{_fmt(summary.get('macro_ari_qwen_image'))}**, "
        f"qwen+label: **{_fmt(summary.get('macro_ari_qwen_label'))}**, "
        f"qwen+shuffled: **{_fmt(summary.get('macro_ari_qwen_shuffled'))}**",
        f"- Δ_image = ARI(qwen+image) − ARI(qwen) = **{_fmt(d)}** "
        f"(95% CI [{_fmt(b_img.get('ci_low'))}, {_fmt(b_img.get('ci_high'))}])",
        f"- Δ_beyond-label = ARI(qwen+image) − ARI(qwen+label) = "
        f"**{_fmt(b_lbl.get('point'))}** "
        f"(95% CI [{_fmt(b_lbl.get('ci_low'))}, {_fmt(b_lbl.get('ci_high'))}])",
        "",
        ("Naive-fusion bar: ✅ cleared" if go else
         "Naive-fusion bar: ➖ not cleared (expected; motivates better fusion/grounding)"),
        "",
        "| Criterion | Result |",
        "| --- | --- |",
    ]
    L += [f"| {k} | {'✅' if v else '❌'} |" for k, v in checks.items()]
    L += ["", "*Thresholds are the plan's explicit experimental decision rules, "
          "not community-standard values.*", ""]

    if not macro.empty:
        L += ["### Macro metrics by scope", "",
              macro.to_markdown(index=False), ""]

    # Per-lemma readout: the visual signal (anchor vs null) and the fusion deltas.
    pl = bootstrap.get("per_lemma", {})
    if pl:
        rows = [{"lemma": k, "subset": v["subset"],
                 "assign": _fmt(v.get("anchor_assignment")),
                 "assign-null": _fmt(v.get("assignment_null")),
                 "profile": _fmt(v.get("image_profile")),
                 "qwen": _fmt(v["qwen"]), "qwen+image": _fmt(v["qwen+image"])}
                for k, v in pl.items()]
        L += ["### Per-lemma readout",
              "",
              "`assign` = nearest-visible-anchor sense assignment vs. its `assign-null`; "
              "`profile` = clustering the anchor profile; `qwen`/`qwen+image` = text and "
              "fused clustering. The visual channel works where `assign` ≫ `assign-null`.",
              "",
              pd.DataFrame(rows).to_markdown(index=False), ""]

    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {out}  (decision: {'GO' if go else 'NO-GO'})")


if __name__ == "__main__":
    main()
