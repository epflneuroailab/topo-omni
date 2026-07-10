#!/usr/bin/env python3
"""Render Fig. A2 — the 3x5 fROI-category x condition response grid (Branch A).

Consumes the per-condition response details CSV produced by
`extract_condition_responses` (columns: subject, parcel_category, parcel_name, hemisphere,
roi_label, condition, modality, even_beta, odd_beta, mean_beta) and renders the paper's
Fig. A2 (`replication_of_fig_4_from_Marvi_2025_with_indiv.svg/.png`): a 5-row x 3-column panel grid of
cross-validated per-condition betas, group mean +/- SEM with individual-subject overlays.

Layout (dev `create_figure4_plot`):
  Row 1: FFA, OFA, fSTS      Row 2: PPA, OPA, RSC      Row 3: EBA, VWFA, LOC
  Row 4: Language, Speech, rTPJ                        Row 5: Frontal MD, Legend, Parietal MD
Aggregate panels (per subject x condition means):
  Language   = mean(ifg, mfg, anttemp, posttemp, ag)
  Frontal MD = mean(supfrontal, midfrontal, medialfrontal, ifgop)
  Parietal MD= mean(parietal, precentral)
Each panel baseline-shifts its minimum-condition mean to zero (Badr style); the y tick
formatter adds the baseline back so tick labels read true beta. Fixed condition order:
faces, bodies, scenes, objects, words_scr_objects | false_belief, false_photo, nonwords,
quilted_speech, math. Solid modality line = the ROI's own modality, dashed = the orthogonal.

PORT NOTES vs dev `src/batch_extract_condition_responses.py` (@ ef1da34):
  * Faithful port of `create_figure4_plot` + its condition name/color maps and the inline
    legend panel. The dev per-ROI (`plot_single_roi_badr_style`), 2-bar contrast variant
    (`create_figure4_contrast_plot`), and standalone legend are NOT ported — Fig. A2 is
    `create_figure4_plot` (the 3x5 grid).
  * Split out as a CSV-consuming step: the numeric golden master lives upstream
    (extract_condition_responses -> condition_responses_details CSV, tight-tolerance
    golden); the render is NOT golden-mastered (render-dependent — docs/DESIGN.md §6 / README
    §6b). This module re-plots that CSV.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `Marvi_2025/` importable so `config` resolves when run as a script.
_DATASET_DIR = Path(__file__).resolve().parent.parent
if str(_DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASET_DIR))

# Fixed order + display names/colors (verbatim from the dev plotter).
ORDERED_CONDITIONS = [
    "faces", "bodies", "scenes", "objects", "words_scr_objects",
    "false_belief", "false_photo", "nonwords", "quilted_speech", "math",
]

CONDITION_NAMES = {
    "faces": "Faces", "bodies": "Bodies", "scenes": "Scenes", "objects": "Objects",
    "words_scr_objects": "Words", "false_belief": "False Belief",
    "false_photo": "False Photo", "nonwords": "Non-words",
    "quilted_speech": "Quilted Audio", "math": "Arithmetic",
}

CONDITION_COLORS = {
    "faces": (230/255, 75/255, 53/255), "bodies": (243/255, 155/255, 47/255),
    "scenes": (241/255, 194/255, 50/255), "objects": (239/255, 201/255, 76/255),
    "words_scr_objects": (212/255, 172/255, 13/255),
    "false_belief": (90/255, 180/255, 172/255), "false_photo": (76/255, 159/255, 155/255),
    "nonwords": (63/255, 143/255, 156/255), "quilted_speech": (59/255, 91/255, 146/255),
    "math": (47/255, 62/255, 117/255),
}

# 3x5 panel layout + display names (verbatim from the dev plotter).
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
VISUAL_ROIS = ["lh_ffa", "lh_ofa", "lh_sts", "lh_ppa", "lh_opa", "lh_rsc",
               "lh_eba", "lh_vwfa", "lh_loc"]
AUDITORY_ROIS = ["language", "speech", "lh_tpj", "frontal_md", "parietal_md"]


def _aggregate_category(df, substrings, roi_label):
    """Per subject x condition mean over ROIs whose label contains any of `substrings`."""
    rois = [r for r in df["roi_label"].unique() if any(p in r for p in substrings)]
    agg = df[df["roi_label"].isin(rois)].groupby(["subject", "condition"])["mean_beta"].mean().reset_index()
    agg["roi_label"] = roi_label
    return agg


def _draw_legend_panel(ax, plt):
    ax.axis("off")
    ax.text(0.05, 0.97, "Visual", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    y = 0.79
    for cond in ORDERED_CONDITIONS[:5]:
        ax.add_patch(plt.Rectangle((0.05, y), 0.08, 0.08, facecolor=CONDITION_COLORS[cond],
                                   edgecolor="black", linewidth=0.5, transform=ax.transAxes))
        ax.text(0.15, y + 0.04, CONDITION_NAMES[cond], transform=ax.transAxes, va="center", fontsize=9)
        y -= 0.14
    ax.text(0.52, 0.97, "Auditory", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    y = 0.79
    for cond in ORDERED_CONDITIONS[5:]:
        ax.add_patch(plt.Rectangle((0.52, y), 0.08, 0.08, facecolor=CONDITION_COLORS[cond],
                                   edgecolor="black", linewidth=0.5, transform=ax.transAxes))
        ax.text(0.62, y + 0.04, CONDITION_NAMES[cond], transform=ax.transAxes, va="center", fontsize=9)
        y -= 0.14
    yl = 0.10
    ax.plot([0.05, 0.25], [yl, yl], "k-", linewidth=2.5, transform=ax.transAxes)
    ax.plot([0.25, 0.45], [yl, yl], "k--", linewidth=2.5, transform=ax.transAxes)
    ax.text(0.47, yl + 0.02, "average beta weights in", transform=ax.transAxes, va="center", fontsize=8)
    ax.text(0.47, yl - 0.04, "orthogonal modality", transform=ax.transAxes, va="center", fontsize=8)
    ax.set_title("Conditions", fontsize=11, fontweight="bold")


def create_figure_a2(df, output_dir, show_individual_data: bool = True):
    """Faithful port of dev `create_figure4_plot`. Writes replication_of_fig_4_from_Marvi_2025{suffix}.svg/.png.

    `df` needs columns roi_label, condition, subject, mean_beta. Returns the SVG Path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(context="paper", font_scale=2, style="white")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    # Aggregate ROIs (per subject x condition), appended to the per-ROI data.
    df_agg = pd.concat([
        df.copy(),
        _aggregate_category(df, ["ifg", "mfg", "anttemp", "posttemp", "ag"], "language"),
        _aggregate_category(df, ["supfrontal", "midfrontal", "medialfrontal", "ifgop"], "frontal_md"),
        _aggregate_category(df, ["parietal", "precentral"], "parietal_md"),
    ], ignore_index=True)

    fig, axes = plt.subplots(5, 3, figsize=(24, 20))
    for row_idx, row_rois in enumerate(ROI_LAYOUT):
        for col_idx, roi in enumerate(row_rois):
            ax = axes[row_idx, col_idx]
            if roi == "legend":
                _draw_legend_panel(ax, plt)
                continue

            roi_df = df_agg[df_agg["roi_label"] == roi].copy()
            if len(roi_df) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            group_stats = roi_df.groupby("condition")["mean_beta"].agg(
                ["mean", "sem", "count"]).reindex(ORDERED_CONDITIONS)
            n_subjects = group_stats["count"].iloc[0] if not group_stats.empty else 1

            # Baseline-shift the minimum-condition mean to zero (Badr style).
            baseline = group_stats["mean"].min()
            group_stats = group_stats.copy()
            group_stats["mean"] -= baseline
            roi_df["mean_beta"] -= baseline

            x_pos = np.concatenate([np.arange(5), np.arange(5) + 6])  # gap between modalities
            colors = [CONDITION_COLORS.get(c, (0.5, 0.5, 0.5)) for c in ORDERED_CONDITIONS]

            if n_subjects > 1:
                ax.bar(x_pos, group_stats["mean"].values, yerr=group_stats["sem"].values,
                       color=colors, alpha=0.9, zorder=2, edgecolor="black", linewidth=1,
                       error_kw={"elinewidth": 1, "capsize": 0})
            else:
                ax.bar(x_pos, group_stats["mean"].values, color=colors, alpha=0.9,
                       zorder=2, edgecolor="black", linewidth=1)

            if n_subjects > 1 and show_individual_data:
                for subj in roi_df["subject"].unique():
                    subj_data = roi_df[roi_df["subject"] == subj].set_index("condition").reindex(ORDERED_CONDITIONS)
                    ax.plot(x_pos[:5], subj_data["mean_beta"].values[:5], "o-", color="gray",
                            alpha=0.35, markersize=3, linewidth=0.8, zorder=3)
                    ax.plot(x_pos[5:], subj_data["mean_beta"].values[5:], "o-", color="gray",
                            alpha=0.35, markersize=3, linewidth=0.8, zorder=3)

            visual_mean = group_stats["mean"].values[:5].mean()
            auditory_mean = group_stats["mean"].values[5:].mean()
            if roi in VISUAL_ROIS:
                ax.hlines(visual_mean, x_pos[0], x_pos[4], colors="gray", linestyles="solid",
                          linewidth=2.5, alpha=0.8, zorder=4)
                ax.hlines(auditory_mean, x_pos[5], x_pos[9], colors="gray", linestyles="dashed",
                          linewidth=2.5, alpha=0.8, zorder=4)
            elif roi in AUDITORY_ROIS:
                ax.hlines(visual_mean, x_pos[0], x_pos[4], colors="gray", linestyles="dashed",
                          linewidth=2.5, alpha=0.8, zorder=4)
                ax.hlines(auditory_mean, x_pos[5], x_pos[9], colors="gray", linestyles="solid",
                          linewidth=2.5, alpha=0.8, zorder=4)

            ax.set_xticks([])
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _, b=baseline: f"{y + b:.2f}"))
            ax.set_ylabel("Beta")
            ax.set_title(ROI_NAMES.get(roi, roi), fontsize=12, fontweight="bold")
            ax.grid(False)
            ax.set_facecolor("white")
            ax.axvline(4.5, color="lightgray", linewidth=2, zorder=1)
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.5)
    suffix = "_with_indiv" if show_individual_data else "_no_indiv"
    stem = f"replication_of_fig_4_from_Marvi_2025{suffix}"
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Fig. A2 -> {svg}  (+ {png.name})")
    return svg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--details-csv", required=True,
                   help="condition_responses_details.csv from extract_condition_responses.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write the figure (default: alongside the details CSV).")
    p.add_argument("--no-individual", action="store_true",
                   help="Omit the individual-subject overlays (group mean/SEM only).")
    return p


def main(argv=None) -> int:
    import pandas as pd

    args = build_parser().parse_args(argv)
    details = Path(args.details_csv)
    if not details.exists():
        print(f"details CSV not found: {details}")
        return 1
    output_dir = Path(args.output_dir) if args.output_dir else details.parent
    df = pd.read_csv(details)
    create_figure_a2(df, output_dir, show_individual_data=not args.no_individual)
    return 0


if __name__ == "__main__":
    sys.exit(main())
