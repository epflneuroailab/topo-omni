"""Fig 2c: per-cluster selectivity scores, one point per discovered cluster.

    python -m src.experiments.spacetop_discovery.plot_model_selectivity \
        --model_name topo-omni-tiled32-4608
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.experiments.spacetop_discovery.common import (
    add_run_args, read_json, resolve_run_args,
)


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-cluster model selectivity scores (Fig 2c).")
    add_run_args(parser)
    parser.add_argument("--scores_path", type=str, default=None,
                        help="Default: <output_dir>/cluster_selectivity_scores_v1.json")
    parser.add_argument("--figures_dir", type=str, default=None,
                        help="Default: <output_dir>/figures")
    parser.add_argument("--brain_clusters", type=str, default="",
                        help="Comma-separated cluster ids that match a known brain "
                             "region, drawn in green. Run-specific -- read them off the "
                             "collages; empty means colour nothing.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Draw a horizontal reference line at this t value.")
    parser.add_argument("--metrics", type=str, default="mean_significant_q=0.001_t",
                        help="Comma-separated metric keys to plot, one figure each.")
    args = resolve_run_args(parser.parse_args())

    scores_path = args.scores_path or os.path.join(
        args.output_dir, "cluster_selectivity_scores_v1.json")
    figures_dir = args.figures_dir or os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    selectivity_scores = read_json(scores_path)
    metrics = [m for m in args.metrics.split(",") if m]
    brain = {int(c) for c in args.brain_clusters.split(",") if c.strip()}

    df = pd.DataFrame([
        {"cluster_id": k, **{metric: v[metric] for metric in metrics}}
        for k, v in selectivity_scores.items()
    ])
    df["is_brain_cluster"] = df["cluster_id"].apply(lambda x: int(int(x) in brain))

    for metric in metrics:
        sns.set_theme(context="paper", font_scale=1.5, style="whitegrid")
        fig, ax = plt.subplots(figsize=(4, 6))
        plot_df = df.melt(id_vars=["is_brain_cluster", "cluster_id"], value_vars=[metric],
                          var_name="metric", value_name="score")
        sns.stripplot(
            data=plot_df, x="metric", y="score", hue="is_brain_cluster",
            palette={0: "red", 1: "green"}, jitter=0.2, size=10, ax=ax,
        )

        # Annotate each point with its cluster_id; match by y-value since x is jittered.
        for collection in ax.collections:
            for x_pos, y_pos in collection.get_offsets():
                match = plot_df[plot_df["score"] == y_pos]
                if not match.empty:
                    ax.annotate(str(match.iloc[0]["cluster_id"]), xy=(x_pos, y_pos),
                                xytext=(6, 0), textcoords="offset points",
                                fontsize=8, va="center")

        if args.threshold is not None:
            ax.axhline(args.threshold, color='gray', linestyle='--', linewidth=1)

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        sns.despine()
        plt.xticks([])
        plt.xlabel("")
        plt.ylabel("Selectivity Score (t-value)")
        ax.set_title(f"Model Selectivity Scores\n{args.model_name}", fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(
            figures_dir, f"selectivity_scores_{metric.replace('=', '')}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Wrote {save_path}")
        plt.close()


if __name__ == "__main__":
    main()
