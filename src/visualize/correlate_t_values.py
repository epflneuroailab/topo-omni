import os
import json 
import numpy as np
import pandas as pd
import scipy.stats as stats

def read_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    # load all json files in the current directory
    path = "results/spacetop_fmri_tvalues.csv"
    fmri_df = pd.read_csv(path, header=0)

    # load all json files in the current directory
    dirpath = "results/qwen2_5_3b_spatial_task_final_7/spacetop_clusters_figures"
    cluster_ids = np.arange(1, 15)
    model_data = []
    for cluster_id in cluster_ids:
        json_path = os.path.join(dirpath, str(cluster_id).zfill(2), "island_morans_I_results_rating_audio=True_top10.json")
        data = read_json(json_path)
        model_data.append({
            "cluster_id": cluster_id,
            "median": data["t_stats"]["median"],
            "mean": data["t_stats"]["mean"],
            "max": data["t_stats"]["max"],
        })

    model_df = pd.DataFrame(model_data)

    # remove cluster 9 from both dataframes
    # model_df = model_df[model_df["cluster_id"] != 1].reset_index()
    # fmri_df = fmri_df[fmri_df["cluster_id"] != 1].reset_index()

    # correlate the t_stats from model_df with the t_stats from fmri_df
    for stat in ["median", "mean", "max"]:
        model_stat = model_df[stat]
        fmri_stat = fmri_df[stat]

        corr, p_value = stats.pearsonr(model_stat, fmri_stat)
        print(f"Correlation between model {stat} and fMRI {stat}: {corr:.4f}")
        print(f"P-value: {p_value:.4f}")

        # plot the correlation
        import seaborn as sns
        import matplotlib.pyplot as plt
        # plot cluster_id label for each data point

        sns.scatterplot(x=model_stat, y=fmri_stat)
        for i in range(len(model_stat)):
            plt.text(model_stat[i], fmri_stat[i], str(model_df["cluster_id"][i]), fontsize=9)
        plt.xlabel(f"Model {stat}")
        plt.ylabel(f"fMRI {stat}")
        plt.title(f"Correlation between model {stat} and fMRI {stat}: {corr:.4f}")
        plt.savefig(f"correlation_{stat}.png")
        plt.close()
        plt.cla()
        plt.clf()
     


    