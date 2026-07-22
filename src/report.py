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
        f"## Decision: {'✅ GO' if go else '⛔ NO-GO'}",
        "",
        f"- Visual lemmas evaluated: **{n_vis}**",
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
        "### Go/no-go criteria",
        "",
        "| Criterion | Result |",
        "| --- | --- |",
    ]
    L += [f"| {k} | {'✅' if v else '❌'} |" for k, v in checks.items()]
    L += ["", "*Thresholds are the plan's explicit experimental decision rules, "
          "not community-standard values.*", ""]

    # Diagnostic interpretation when NO-GO.
    if not go:
        L += ["### Interpretation", ""]
        if abs(summary.get("macro_ari_qwen_image", 0)
               - summary.get("macro_ari_qwen_label", 0)) < 0.01:
            L.append("- **Image ≈ label control:** class *names* explain the effect; "
                     "images add little.")
        if abs(summary.get("macro_ari_qwen_image", 0)
               - summary.get("macro_ari_qwen_shuffled", 0)) < 0.01:
            L.append("- **Image ≈ shuffled control:** the anchor profile is not carrying "
                     "sense-specific information.")
        L.append("")

    if not macro.empty:
        L += ["### Macro metrics by scope", "",
              macro.to_markdown(index=False), ""]

    # Per-lemma image deltas.
    pl = bootstrap.get("per_lemma", {})
    if pl:
        rows = [{"lemma": k, "subset": v["subset"], "qwen": _fmt(v["qwen"]),
                 "qwen+image": _fmt(v["qwen+image"]), "qwen+label": _fmt(v["qwen+label"]),
                 "Δ_image": _fmt(v["delta_image"])} for k, v in pl.items()]
        L += ["### Per-lemma (LOLO-tuned λ)", "",
              pd.DataFrame(rows).to_markdown(index=False), ""]

    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {out}  (decision: {'GO' if go else 'NO-GO'})")


if __name__ == "__main__":
    main()
