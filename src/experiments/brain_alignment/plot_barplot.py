import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

SAVE_DIR = os.getenv("SAVE_DIR")

layer_to_roi = {
    "faces": ["FFA-1", "FFA-2", "OFA"],
    "scenes": ["PPA", "RSC", "OPA"],
    "vwfa": ["VWFA-1", "VWFA-2", "OWFA"],
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

    top_k = 1

    columns_to_keep = ['layer_name', 'subject', 'roi', 'pearsonr_nc', 'model_name']

    path = f"{SAVE_DIR}/nsd_topo_omni_results_merged.csv"
    df = pd.read_csv(path, header=0)

    df = df[df["layer_name"].str.contains(f"top{top_k}")]
    # concat model_name, task_loss, spatial_loss into one column
    df["model_name"] = df["model_name"] + "_" + df["task_loss"].astype(str) + "_" + df["spatial_loss"].astype(str)

    frois = df["roi"].unique().tolist()

    final_df = pd.DataFrame()
    for layer, rois in layer_to_roi.items():
        df_layer = df[
            df["layer_name"].str.contains(layer)
            & df["roi"].isin(rois)
        ]

        df_layer["layer_name"] = layer_to_name.get(layer, layer)
        df_layer["model_name"] = df_layer["model_name"].map(model_name_mapping)
        
        df_layer = df_layer[df_layer["model_name"] != "Topo-Omni (Spatial)"] 

        df_layer = df_layer[columns_to_keep]
        final_df = pd.concat([final_df, df_layer], axis=0)

    sns.set_theme(style="whitegrid", font_scale=1.5, context="paper")
    plt.figure(figsize=(12, 8))

    model_names_order = [
        "Topo-Omni",
        "Qwen2.5-3B (SFT)",
        "Qwen2.5-3B (Baseline)",
    ]

    g = sns.catplot(
        kind="bar",
        data=final_df,
        x="roi",
        y="pearsonr_nc",
        hue="model_name",
        col="layer_name",
        height=4,
        aspect=0.8,
        sharex=False,
        sharey=True,
        palette="Set2",
        hue_order=model_names_order
    )

    sns.despine()
    g.set_axis_labels("ROI", "Pearson's r (Noise Corrected)")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_ylim(0.50, 0.85)

    # remove legend
    g._legend.remove()

    g.add_legend(title=None, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(model_names_order))

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(f"{SAVE_DIR}/brain_alignment_results_top{top_k}.png", dpi=300)



