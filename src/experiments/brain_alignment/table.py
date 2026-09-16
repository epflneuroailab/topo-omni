import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

SAVE_DIR = os.getenv("SAVE_DIR")

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

    # Build LaTeX table
    n_models = len(model_order)
    col_spec = "ll" + "c" * n_models
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Brain alignment results (Pearson's $r$, noise-corrected). Mean $\pm$ std across subjects. \textbf{Bold} indicates best per ROI.}")
    lines.append(r"\label{tab:brain_alignment}")
    # lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header
    header = r"Category & ROI & " + " & ".join(model_order) + r" \\"
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
            row += r" \\"
            lines.append(row)

        # Add midrule between categories (except after last)
        if cat != category_order[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)
    print(latex_str)

    out_path = f"{SAVE_DIR}/brain_alignment_table.tex"
    with open(out_path, "w") as f:
        f.write(latex_str)
    print(f"\nSaved to {out_path}")