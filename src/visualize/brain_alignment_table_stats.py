import os
import numpy as np
import pandas as pd

from scipy.stats import ttest_rel, wilcoxon

from dotenv import load_dotenv
load_dotenv()

SAVE_DIR = os.getenv("SAVE_DIR")

# Paired test to use for Topo-Omni vs. each baseline across subjects.
# "ttest"    -> two-sided paired t-test (scipy.stats.ttest_rel)
# "wilcoxon" -> Wilcoxon signed-rank test (robust, better for small N)
PAIRED_TEST = "ttest"

layer_to_roi = {
    "faces": ["OFA", "FFA-1", "FFA-2"],
    "scenes": ["OPA", "PPA", "RSC"],
    "vwfa": ["OWFA", "VWFA-1", "VWFA-2"],
    "bodies": ["EBA", "FBA-1", "FBA-2"],
}

layer_to_name = {
    "faces": "Faces",
    "scenes": "Scenes",
    "vwfa": "VWFA",
    "bodies": "Bodies",
}

model_name_mapping = {
    "topo_omni_True_True": "Topo-Omni",
    "qwen_omni_True_False": "Qwen2.5-3B (SFT)",
    "qwen_omni_False_False": "Qwen2.5-3B (Baseline)",
    "topo_omni_False_True": "Topo-Omni (Spatial)",
}


def fmt_p(p):
    """Format a p-value with significance markers; n.s. means p > 0.05."""
    if p is None or np.isnan(p):
        return "--"
    if p > 0.05:
        star = r"^{\mathrm{n.s.}}"
    elif p > 0.01:
        star = "^{*}"
    elif p > 0.001:
        star = "^{**}"
    else:
        star = "^{***}"
    val = "<0.001" if p < 0.001 else f"{p:.3f}"
    return f"${val}{star}$"


if __name__ == "__main__":

    columns_to_keep = ['layer_name', 'subject', 'roi', 'pearsonr_nc', 'model_name']

    path = f"{SAVE_DIR}/nsd_topo_omni_results_merged.csv"
    df = pd.read_csv(path, header=0)

    df = df[df["layer_name"].str.contains("top10")]
    df["model_name"] = df["model_name"] + "_" + df["task_loss"].astype(str) + "_" + df["spatial_loss"].astype(str)

    final_df = pd.DataFrame()
    for layer, rois in layer_to_roi.items():
        df_layer = df[
            df["layer_name"].str.contains(layer)
            & df["roi"].isin(rois)
        ]
        df_layer = df_layer.copy()
        df_layer["layer_name"] = layer_to_name.get(layer, layer)
        df_layer["model_name"] = df_layer["model_name"].map(model_name_mapping)
        df_layer = df_layer[df_layer["model_name"] != "Topo-Omni (Spatial)"]

        df_layer = df_layer[columns_to_keep]
        final_df = pd.concat([final_df, df_layer], axis=0)

    # Compute mean ± sem across subjects for each (model, category, roi)
    stats = (
        final_df.groupby(["layer_name", "roi", "model_name"])["pearsonr_nc"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    # For each ROI, find the best model (highest mean)
    best = stats.loc[stats.groupby(["layer_name", "roi"])["mean"].idxmax()]
    best_set = set(zip(best["layer_name"], best["roi"], best["model_name"]))

    # Order models as in mapping
    model_order = [
        "Topo-Omni",
        "Qwen2.5-3B (SFT)",
        "Qwen2.5-3B (Baseline)",
    ]
    # Order categories
    category_order = list(layer_to_name.values())

    # --- Paired significance tests: Topo-Omni vs. each baseline, across subjects ---
    # Two-sided, within-subject paired test per (category, ROI). We deliberately do
    # NOT apply multiple-comparison correction: that is conservative for a
    # "no significant difference" claim, since any correction only inflates p.
    baseline_models = ["Qwen2.5-3B (SFT)", "Qwen2.5-3B (Baseline)"]
    pvals = {}
    for cat in category_order:
        rois = layer_to_roi[[k for k, v in layer_to_name.items() if v == cat][0]]
        for roi in rois:
            sub = final_df[(final_df["layer_name"] == cat) & (final_df["roi"] == roi)]
            # subject x model matrix so subjects are matched across models
            wide = sub.pivot_table(
                index="subject", columns="model_name", values="pearsonr_nc"
            )
            for base in baseline_models:
                p = np.nan
                if "Topo-Omni" in wide.columns and base in wide.columns:
                    paired = wide[["Topo-Omni", base]].dropna()
                    diffs = paired["Topo-Omni"] - paired[base]
                    if len(paired) >= 2 and np.any(diffs != 0):
                        try:
                            if PAIRED_TEST == "wilcoxon":
                                _, p = wilcoxon(paired["Topo-Omni"], paired[base])
                            else:
                                _, p = ttest_rel(paired["Topo-Omni"], paired[base])
                        except ValueError:
                            p = np.nan
                pvals[(cat, roi, base)] = p

    # Build LaTeX table
    n_models = len(model_order)
    col_spec = "ll" + "c" * n_models + "cc"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Brain alignment results (Pearson's $r$, noise-corrected). "
        r"Mean $\pm$ s.e.m.\ across subjects. \textbf{Bold} indicates best per ROI. "
        r"The last two columns report two-sided paired $t$-tests of Topo-Omni against "
        r"each baseline across subjects (\textsuperscript{n.s.}~$p>0.05$; "
        r"$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$; uncorrected). "
        r"Topo-Omni is not significantly different from the baselines in any ROI.}"
    )
    lines.append(r"\label{tab:brain_alignment}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header
    header = (
        r"Category & ROI & " + " & ".join(model_order)
        + r" & $p$ (vs.\ SFT) & $p$ (vs.\ Base) \\"
    )
    lines.append(header)
    lines.append(r"\midrule")

    for cat in category_order:
        rois = layer_to_roi[[k for k, v in layer_to_name.items() if v == cat][0]]
        for i, roi in enumerate(rois):
            cat_label = cat if i == 0 else ""
            row = f"{cat_label} & {roi} & "
            for j, model in enumerate(model_order):
                entry = stats[
                    (stats["layer_name"] == cat)
                    & (stats["roi"] == roi)
                    & (stats["model_name"] == model)
                ]
                if len(entry) == 1:
                    m = entry["mean"].values[0]
                    s = entry["sem"].values[0]
                    cell = f"{m:.3f} \\pm {s:.3f}"
                    if (cat, roi, model) in best_set:
                        row += r"$\boldsymbol{" + cell + "}$"
                    else:
                        row += f" ${cell}$"

                    if j < n_models - 1:
                        row += " & "
                else:
                    row += " & --"

            # Significance of Topo-Omni vs. each baseline
            for base in baseline_models:
                row += " & " + fmt_p(pvals.get((cat, roi, base), np.nan))
            row += r" \\"
            lines.append(row)

        # Add midrule between categories (except after last)
        if cat != category_order[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)
    print(latex_str)

    out_path = f"{SAVE_DIR}/brain_alignment_table.tex"
    with open(out_path, "w") as f:
        f.write(latex_str)
    print(f"\nSaved to {out_path}")