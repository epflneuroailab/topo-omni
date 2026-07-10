#!/usr/bin/env python3
"""Fig. B3b — island Moran's I comparison bar chart (step 06).

Three bars: Non-Topo (SFT) | Topo-Omni | Real brain (Pernet 2015). The two model bars
show mean ± SE across the model's per-island Moran's I values; the brain bar is the
point estimate (mean island Moran's I of the voice-selective map, from step 05).

Significance (both reported, annotations use the parametric test — matches the dev fig):
  * parametric     : one-sample t-test  — model island distribution vs the brain mean
                     (one-sided: brain > model; any model is incomplete → lower I)
  * non-parametric : Wilcoxon signed-rank — model distribution vs the brain mean

Lineage (docs/DESIGN.md §2.4):  02 surface → 05 island Moran's I → **06 comparison plot**.
  input : <results-root>/03_spatial_analysis/island_morans_i_results.json   (from 05)
          + data/model_island_morans_i/{topoomni,nontopo}_vocals_fwhm4.0_island_morans_I.json
  output: <results-root>/03_spatial_analysis/island_morans_i_comparison.{svg,png,pdf}

PORT NOTES vs dev-repo `src/06_plot_island_morans_i_comparison.py` (@ f842b1a):
  * The model bars now read the **vendored per-island distributions** in
    `data/model_island_morans_i/` (was an absolute path into the Marvi repo; see
    data/PROVENANCE.md for the sha256 pins). These are the exact artifacts the published
    figure was drawn from — Topo-Omni mean I≈0.575 (n=79 islands), Non-Topo I≈0.235
    (n=418). They **supersede** the orphaned point-estimate literals in the old
    `data/fig_b3b_model_island_morans_i.json` (0.594 / 0.126), which came from a
    non-figure computation in dev `05` and never matched this chart (docs/DESIGN.md §7/§10).
  * Parameterized by `--results-root` (was hard-coded `results/...`); brain values read
    from step 05's JSON output.
  * Model bars are vendored now, to be swapped for a live topo-omni run at merge
    (docs/DESIGN.md §10). The chart is a visual artefact and is not golden-mastered; the
    reproducible numeric net is the model mean/SE per bar (tests/test_b3b_comparison_*).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon

_HERE = Path(__file__).resolve().parent
_DATASET = _HERE.parent      # Pernet_2015/
MODEL_DIR = _DATASET / "data" / "model_island_morans_i"
LOCALIZER = "vocals"
FWHM = 4.0
# (model name in the vendored filenames, x-axis label, RGB colour) — dev order/colours.
MODELS = [
    ("nontopo", "Non-Topo", (16 / 255, 67 / 255, 200 / 255)),
    ("topoomni", "Topo-Omni", (0 / 255, 127 / 255, 255 / 255)),
]
COLOR_BRAIN = (77 / 255, 153 / 255, 0 / 255)


def load_model_islands(model_name: str, model_dir: Path = MODEL_DIR) -> np.ndarray:
    """Per-island Moran's I values for a model (vendored distribution)."""
    fname = Path(model_dir) / f"{model_name}_{LOCALIZER}_fwhm{FWHM}_island_morans_I.json"
    if not fname.exists():
        raise FileNotFoundError(f"Vendored model per-island JSON not found: {fname}")
    d = json.loads(fname.read_text())
    return np.array([v["moran_I"] for v in d.values()], dtype=float)


def load_brain(results_root: Path):
    """Brain point estimate + per-island values from step 05's JSON."""
    brain_json = Path(results_root) / "03_spatial_analysis" / "island_morans_i_results.json"
    if not brain_json.exists():
        raise FileNotFoundError(
            f"Brain results not found: {brain_json}. Run 05_island_morans_i.py first."
        )
    d = json.loads(brain_json.read_text())
    return float(d["avg_I_unweighted"]), np.array(d["all_island_I_values"], dtype=float)


def compute_bars(results_root: Path, model_dir: Path = MODEL_DIR) -> dict:
    """Assemble the bar heights + SEs + brain-vs-model stats. Side-effect-free (testable)."""
    brain_mean, brain_vals = load_brain(results_root)
    bars = {"brain": {"mean": brain_mean, "se": None, "n": int(brain_vals.size)}}
    stats = {}
    for name, label, _ in MODELS:
        vals = load_model_islands(name, model_dir)
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        # One-sided: brain > model, i.e. the model distribution lies below the brain mean.
        t, p_t = ttest_1samp(vals, brain_mean, alternative="less")
        _, p_w = wilcoxon(vals - brain_mean, alternative="less")
        bars[name] = {"mean": mean, "se": se, "n": int(vals.size), "label": label}
        stats[name] = {"t": float(t), "p_ttest": float(p_t), "p_wilcoxon": float(p_w)}
    return {"bars": bars, "stats": stats, "brain_mean": brain_mean}


def _sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def plot_comparison(result: dict, out_dir: Path):
    """Render the 3-bar Fig. B3b chart. Not golden-mastered (visual artefact)."""
    import matplotlib
    matplotlib.use("Agg")           # headless
    import matplotlib.pyplot as plt

    bars, stats = result["bars"], result["stats"]
    brain_mean = result["brain_mean"]

    BAR_WIDTH = 0.275
    POS = [0.0, 0.45, 0.90]         # non-topo, topo, brain
    fig, ax = plt.subplots(figsize=(4.2, 4.5))

    # Model bars (mean ± SE), in dev order.
    for pos, (name, label, color) in zip(POS[:2], MODELS):
        b = bars[name]
        ax.bar(pos, b["mean"], width=BAR_WIDTH, color=color, edgecolor="none", zorder=2)
        ax.errorbar(pos, b["mean"], yerr=b["se"], fmt="none", color="black",
                    capsize=4, linewidth=1.5, zorder=3)

    # Brain bar — point estimate, no error bar.
    ax.bar(POS[2], brain_mean, width=BAR_WIDTH, color=COLOR_BRAIN, edgecolor="none", zorder=2)

    # Significance brackets vs the brain (parametric p, matching the dev annotation).
    tops = [bars["nontopo"]["mean"] + bars["nontopo"]["se"],
            bars["topoomni"]["mean"] + bars["topoomni"]["se"], brain_mean]
    b0 = max(tops) + 0.02
    b1 = b0 + 0.04

    def bracket(x1, x2, y, label):
        h = 0.015
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=0.9)
        ax.text((x1 + x2) / 2, y + h + 0.004, label, ha="center", va="bottom", fontsize=9)

    bracket(POS[1], POS[2], b0, _sig_label(stats["topoomni"]["p_ttest"]))
    bracket(POS[0], POS[2], b1, _sig_label(stats["nontopo"]["p_ttest"]))

    ax.set_xticks(POS)
    ax.set_xticklabels(["Non-Topo", "Topo-Omni", "Brain\n(Pernet 2015)"], fontsize=13)
    ax.set_ylabel("Island Moran's I", fontsize=13)
    ax.set_ylim(bottom=0, top=0.7)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("svg", "png", "pdf"):
        out = out_dir / f"island_morans_i_comparison.{suffix}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        written.append(out)
    plt.close(fig)
    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, required=True,
                   help="Root holding 03_spatial_analysis/island_morans_i_results.json (from 05); output too.")
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                   help="Vendored model per-island Moran's I JSONs (Fig. B3b model bars).")
    p.add_argument("--no-figure", action="store_true", help="Print stats only; skip the render.")
    p.add_argument("--plots-dir", type=Path, default=None,
                   help="Directory for island_morans_i_comparison.{svg,png,pdf} "
                        "(default: <results-root>/03_spatial_analysis).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = compute_bars(args.results_root, args.model_dir)
    bars, stats = result["bars"], result["stats"]

    print(f"Brain (point estimate): {result['brain_mean']:.4f}  (n={bars['brain']['n']} islands)")
    for name, label, _ in MODELS:
        b, s = bars[name], stats[name]
        print(f"{label:10s} mean ± SE: {b['mean']:.4f} ± {b['se']:.4f}  (n={b['n']} islands)  "
              f"vs brain: t={s['t']:.3f}, p_t={s['p_ttest']:.4g} {_sig_label(s['p_ttest'])}, "
              f"p_wilcoxon={s['p_wilcoxon']:.4g} {_sig_label(s['p_wilcoxon'])}")

    if not args.no_figure:
        out_dir = args.plots_dir or (args.results_root / "03_spatial_analysis")
        for f in plot_comparison(result, out_dir):
            print(f"Fig. B3b -> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
