import os
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot model selectivity scores for each cluster.")
    parser.add_argument("--model_name", type=str, default="qwen2_5_3b_spatial_task_final_7", help="Model name to evaluate")
    args = parser.parse_args()

    model_name = args.model_name

    selectivity_scores_path = f"task-alignvideo/clustering_v2/cluster_selectivity_scores_v1.json"

    with open(selectivity_scores_path, 'r') as f:
        selectivity_scores = json.load(f)

    metrics = [
        # "mean_t", 
        # "mean_significant_q=0.05_t",
        # "mean_significant_q=0.01_t",
        "mean_significant_q=0.001_t",
        # "median_t",
        # "median_significant_q=0.05_t",
        # "median_significant_q=0.01_t", 
        # "median_significant_q=0.001_t"
    ]

    # Convert the selectivity scores to a DataFrame for plotting
    df = pd.DataFrame([
        {"cluster_id": k, **{metric: v[metric] for metric in metrics}}
        for k, v in selectivity_scores.items()
    ])

    df["is_brain_cluster"] = df["cluster_id"].apply(lambda x: int(x) in [5,6,7,30,31,32,49])

    # remove cluster 14
    # df = df[df["cluster_id"] != "14"]

    for metric in metrics:

        # plot scatter plot of median_t in one x-tick and median_significant_t in another x-tick, with is_brain_cluster as hue
        sns.set_theme(context="paper", font_scale=1.5, style="whitegrid")
        fig, ax = plt.subplots(figsize=(4, 6))
        plot_df = df.melt(id_vars=["is_brain_cluster", "cluster_id"], value_vars=[metric], var_name="metric", value_name="score")
        sns.stripplot(
            data=plot_df,
            x="metric",
            y="score",
            hue="is_brain_cluster",
            palette={0: "red", 1: "green"},
            jitter=0.2,
            size=10,
            ax=ax
        )

        # Annotate each point with its cluster_id; match by y-value since x is jittered
        for collection in ax.collections:
            for x_pos, y_pos in collection.get_offsets():
                match = plot_df[plot_df["score"] == y_pos]
                if not match.empty:
                    ax.annotate(str(match.iloc[0]["cluster_id"]), xy=(x_pos, y_pos),
                                xytext=(6, 0), textcoords="offset points",
                                fontsize=8, va="center")

        if metric == "mean_significant_q=0.001_t":
            ax.axhline(18.5, color='gray', linestyle='--', linewidth=1)

        # remove legend
        ax.get_legend().remove()
        sns.despine()
        plt.xticks([])
        plt.xlabel("")
        plt.ylabel("Selectivity Score (t-value)")
        ax.set_title(f"Model Selectivity Scores for Topo-Omni")
        # plt.legend(title="Cluster in Brain", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"topoomni_selectivity_scores_v1_{metric}.png", dpi=300, bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()
