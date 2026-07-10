#!/usr/bin/env python
"""Group-level 2-bar contrast plots — the bars beside the main-text native-surface figures.

Faithful port of dev `src/batch_extract_condition_responses.py::create_figure4_contrast_plot`
(+ `_draw_contrast_panel`) — the group-level per-fROI **contrast bars** that accompany each
Branch-B surface map in the paper's main-text figures. Each panel = 2 bars (the two sides of
a contrast, e.g. FFA: Faces vs Objects), **group mean ± SEM across the subjects** (per-subject
mean-in-fROI first, then between-subject SEM), a **paired t-test** with significance stars, and
optional per-subject overlay lines.

Same numeric source as Fig. A2 (`plot_figure_a2`): the `condition_responses_details*.csv`
produced by `extract_condition_responses` — NO new GLM/extraction (it's a second renderer over
the same DataFrame). Group-pooled ROIs (Language, Frontal/Parietal MD) average their member
parcels per subject first, exactly as the dev grid does.

Outputs (into --output-dir):
  - `group_contrast_bars{_with_indiv|_no_indiv}.{svg,png}`  — combined 5x3 grid (all 14 fROIs)
  - `group_contrast_bars/<roi>{suffix}.{svg,png}`            — one file per fROI

This is a RENDER step (not golden-mastered, docs/DESIGN.md §6); the numeric golden is upstream
(`condition_responses_details`, tight tolerance).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Condition -> RGB (mirrors plot_figure_a2.CONDITION_COLORS; kept local to avoid cross-module
# import fragility when this runs standalone vs as part of the `analysis` package).
CONDITION_COLORS = {
    "faces": (230/255, 75/255, 53/255), "bodies": (243/255, 155/255, 47/255),
    "scenes": (241/255, 194/255, 50/255), "objects": (239/255, 201/255, 76/255),
    "words_scr_objects": (212/255, 172/255, 13/255),
    "false_belief": (90/255, 180/255, 172/255), "false_photo": (76/255, 159/255, 155/255),
    "nonwords": (63/255, 143/255, 156/255), "quilted_speech": (59/255, 91/255, 146/255),
    "math": (47/255, 62/255, 117/255),
}

# (group1_conditions, group2_conditions, x-label1, x-label2) per fROI — verbatim from dev.
CONTRAST_GROUPS = {
    "lh_ffa":      (["faces"],                       ["objects"],                    "Faces",   "Objects"),
    "lh_ofa":      (["faces"],                       ["objects"],                    "Faces",   "Objects"),
    "lh_sts":      (["faces"],                       ["objects"],                    "Faces",   "Objects"),
    "lh_ppa":      (["scenes"],                      ["objects"],                    "Scenes",  "Objects"),
    "lh_opa":      (["scenes"],                      ["objects"],                    "Scenes",  "Objects"),
    "lh_rsc":      (["scenes"],                      ["objects"],                    "Scenes",  "Objects"),
    "lh_eba":      (["bodies"],                      ["objects"],                    "Bodies",  "Objects"),
    "lh_vwfa":     (["words_scr_objects"],           ["objects"],                    "Words",   "Objects"),
    "lh_loc":      (["objects"],                     ["words_scr_objects"],          "Objects", "Words"),
    "language":    (["false_belief", "false_photo"], ["nonwords"],                   "FB+FP",   "NW"),
    "speech":      (["nonwords"],                    ["quilted_speech"],             "NW",      "QLT"),
    "lh_tpj":      (["false_belief"],                ["false_photo"],                "FB",      "FP"),
    "frontal_md":  (["math"],                        ["false_belief", "false_photo"],"MATH",    "FB+FP"),
    "parietal_md": (["math"],                        ["false_belief", "false_photo"],"MATH",    "FB+FP"),
}

ROI_LAYOUT = [
    ["lh_ffa", "lh_ofa", "lh_sts"],
    ["lh_ppa", "lh_opa", "lh_rsc"],
    ["lh_eba", "lh_vwfa", "lh_loc"],
    ["language", "speech", "lh_tpj"],
    ["frontal_md", "legend", "parietal_md"],
]

ROI_NAMES = {
    "lh_ffa": "FFA", "lh_ofa": "OFA", "lh_sts": "fSTS",
    "lh_ppa": "PPA", "lh_opa": "OPA", "lh_rsc": "RSC",
    "lh_eba": "EBA", "lh_vwfa": "VWFA", "lh_loc": "LOC",
    "language": "Language", "speech": "Speech", "lh_tpj": "rTPJ",
    "frontal_md": "Frontal MD", "parietal_md": "Parietal MD",
}


def _blend_colors(conditions):
    colors = [CONDITION_COLORS[c] for c in conditions if c in CONDITION_COLORS]
    return tuple(np.mean(colors, axis=0)) if colors else (0.5, 0.5, 0.5)


def _draw_contrast_panel(ax, roi_df, g1_conds, g2_conds, label1, label2, roi_name,
                         show_individual_data, from_zero=False, show_stats=True,
                         show_title=True, gray_grid=False):
    """Draw a 2-bar contrast panel onto ax (verbatim dev logic). Returns the baseline used."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from scipy import stats

    if len(roi_df) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return 0.0

    subjects = roi_df["subject"].unique()
    g1_per_subj, g2_per_subj = [], []
    for subj in subjects:
        subj_df = roi_df[roi_df["subject"] == subj]
        g1_per_subj.append(subj_df[subj_df["condition"].isin(g1_conds)]["mean_beta"].mean())
        g2_per_subj.append(subj_df[subj_df["condition"].isin(g2_conds)]["mean_beta"].mean())
    g1 = np.array(g1_per_subj)
    g2 = np.array(g2_per_subj)

    if from_zero:
        baseline = 0.0
    else:
        baseline = min(np.nanmean(g1), np.nanmean(g2))
        g1 = g1 - baseline
        g2 = g2 - baseline

    n = np.sum(~np.isnan(g1))
    g1_mean, g2_mean = np.nanmean(g1), np.nanmean(g2)
    g1_sem = np.nanstd(g1) / np.sqrt(n) if n > 1 else 0
    g2_sem = np.nanstd(g2) / np.sqrt(n) if n > 1 else 0

    color1, color2 = _blend_colors(g1_conds), _blend_colors(g2_conds)

    if n > 1:
        ax.bar([0, 1], [g1_mean, g2_mean], width=0.5, yerr=[g1_sem, g2_sem],
               color=[color1, color2], alpha=0.9, zorder=2, edgecolor="black", linewidth=1,
               error_kw={"elinewidth": 1, "capsize": 0})
    else:
        ax.bar([0, 1], [g1_mean, g2_mean], width=0.5, color=[color1, color2], alpha=0.9,
               zorder=2, edgecolor="black", linewidth=1)

    if n > 1 and show_individual_data:
        for v1, v2 in zip(g1, g2):
            ax.plot([0, 1], [v1, v2], "o-", color="gray", alpha=0.35, markersize=3,
                    linewidth=0.8, zorder=3)

    if from_zero:
        min_val = min(g1_mean, g2_mean)
        ax.set_ylim(bottom=min_val * 1.1 if min_val < 0 else 0)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([label1, label2])
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _, b=baseline: f"{y + b:.2f}"))
    ax.set_ylabel("Beta")
    if show_title:
        ax.set_title(roi_name, fontsize=12, fontweight="bold")
    if gray_grid:
        ax.set_facecolor("white")
        ax.yaxis.grid(True, color="#cccccc", linewidth=0.8, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
        ax.set_facecolor("white")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if not show_stats:
        return baseline
    valid = ~(np.isnan(g1) | np.isnan(g2))
    if valid.sum() >= 2:
        _, pval = stats.ttest_rel(g1[valid], g2[valid])
        sig_label = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        bar_top = max(g1_mean + g1_sem, g2_mean + g2_sem)
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        bar_y = bar_top + y_range * 0.05
        tick_h = y_range * 0.02
        ax.plot([0, 0, 1, 1], [bar_y - tick_h, bar_y, bar_y, bar_y - tick_h], color="black", linewidth=1)
        ax.text(0.5, bar_y + tick_h * 0.5, sig_label, ha="center", va="bottom",
                fontsize=10 if sig_label == "n.s." else 12)
        ax.set_ylim(top=bar_y + y_range * 0.12)
    return baseline


def _aggregate_group_rois(df):
    """Add pooled Language / Frontal-MD / Parietal-MD ROIs (per-subject parcel means), dev logic."""
    def pool(patterns, new_label):
        rois = [r for r in df["roi_label"].unique() if any(p in r for p in patterns)]
        sub = df[df["roi_label"].isin(rois)].groupby(["subject", "condition"])["mean_beta"].mean().reset_index()
        sub["roi_label"] = new_label
        return sub
    return pd.concat([
        df,
        pool(["ifg", "mfg", "anttemp", "posttemp", "ag"], "language"),
        pool(["supfrontal", "midfrontal", "medialfrontal", "ifgop"], "frontal_md"),
        pool(["parietal", "precentral"], "parietal_md"),
    ], ignore_index=True)


def create_group_contrast_bars(df, output_dir, show_individual_data=True, from_zero=True,
                               show_stats=True, show_title=True, gray_grid=False):
    """Combined 5x3 grid + per-fROI 2-bar group contrast plots (dev create_figure4_contrast_plot).

    ``from_zero=True`` (release default): both bars are drawn from 0 as absolute group-mean betas
    — the intuitive reading. (The dev default `from_zero=False` subtracts min(mean_A, mean_B) from
    both groups, which collapses the lower bar to a flat line and only shows the *difference*; that
    reads as a "missing" second bar and is not what we want in the release.)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(context="paper", font_scale=2, style="white")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    output_dir = Path(output_dir)
    df_agg = _aggregate_group_rois(df)
    suffix = "_with_indiv" if show_individual_data else "_no_indiv"

    # --- per-fROI files ---
    per_roi_dir = output_dir / "group_contrast_bars"
    per_roi_dir.mkdir(parents=True, exist_ok=True)
    all_rois = [roi for row in ROI_LAYOUT for roi in row if roi != "legend"]
    for roi in all_rois:
        if roi not in CONTRAST_GROUPS:
            continue
        g1, g2, l1, l2 = CONTRAST_GROUPS[roi]
        fig_s, ax_s = plt.subplots(figsize=(4, 4))
        fig_s.patch.set_facecolor("white")
        _draw_contrast_panel(ax_s, df_agg[df_agg["roi_label"] == roi], g1, g2, l1, l2,
                             ROI_NAMES.get(roi, roi), show_individual_data, from_zero=from_zero,
                             show_stats=show_stats, show_title=show_title, gray_grid=gray_grid)
        fig_s.tight_layout()
        fig_s.savefig(per_roi_dir / f"{roi}{suffix}.png", dpi=300, bbox_inches="tight")
        fig_s.savefig(per_roi_dir / f"{roi}{suffix}.svg", format="svg", bbox_inches="tight")
        plt.close(fig_s)

    # --- combined grid ---
    fig, axes = plt.subplots(5, 3, figsize=(24, 20))
    for r, row_rois in enumerate(ROI_LAYOUT):
        for c, roi in enumerate(row_rois):
            ax = axes[r, c]
            if roi == "legend":
                ax.axis("off")
                ax.text(0.5, 0.5, "Contrast\nGroups", ha="center", va="center",
                        transform=ax.transAxes, fontsize=14, fontweight="bold")
                continue
            if roi not in CONTRAST_GROUPS:
                ax.text(0.5, 0.5, "No contrast defined", ha="center", va="center", transform=ax.transAxes)
                continue
            g1, g2, l1, l2 = CONTRAST_GROUPS[roi]
            _draw_contrast_panel(ax, df_agg[df_agg["roi_label"] == roi], g1, g2, l1, l2,
                                 ROI_NAMES.get(roi, roi), show_individual_data, from_zero=from_zero,
                                 show_title=show_title, gray_grid=gray_grid)
    plt.tight_layout(pad=1.5)
    combined = output_dir / f"group_contrast_bars{suffix}"
    plt.savefig(f"{combined}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{combined}.svg", format="svg", bbox_inches="tight")
    plt.close()
    print(f"Group contrast bars -> {combined}.svg  (+ per-fROI in {per_roi_dir}/)")
    return Path(f"{combined}.svg")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--details-csv", required=True, help="condition_responses_details*.csv (from extract_condition_responses).")
    p.add_argument("--output-dir", required=True, help="Where to write the group contrast bars.")
    p.add_argument("--no-individual", action="store_true", help="Hide per-subject overlay lines.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    df = pd.read_csv(args.details_csv)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    create_group_contrast_bars(df, Path(args.output_dir), show_individual_data=not args.no_individual)
    return 0


if __name__ == "__main__":
    sys.exit(main())
