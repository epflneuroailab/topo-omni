import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

SAVE_DIR = os.getenv("SAVE_DIR")

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

color_palette = sns.color_palette(["#E64B35", "#F39B2F", "#F1C232", "#EFC94C"])

if __name__ == "__main__":

    model_name = "topo-omni"
    dirpath = f"{SAVE_DIR}/{model_name}/ablation"
    path_template = "similarity_ablation_results_top{percentage}_stimulate=True_v3.json"

    percentages = [5, 10, 15, 20, 25, 30]

    results = []

    for percentage in percentages:
        path = path_template.format(percentage=percentage)
        path = f"{dirpath}/{path}"
        if not os.path.exists(path):
            print(f"> File {path} does not exist. Skipping.")
            continue

        data = read_json(path)
        df = pd.DataFrame(data)
        df["percentage"] = percentage
        df["perceived"] = df.apply(lambda row: row["localizer"] == row["prediction"], axis=1)
        results.append(df)

    results = pd.concat(results, axis=0)
    results["localizer"] = results["localizer"].str.capitalize()
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=2)

    plt.figure(figsize=(4, 4))

    sns.lineplot(
        data=results,
        x="percentage",
        y="perceived",
        hue="localizer",
        marker="o",
        dashes=False,
        markersize=15,
        palette=color_palette,
    )

    sns.despine()

    # remove legend
    plt.legend([], frameon=False)

    plt.xlabel("Coverage (%)")
    plt.ylabel("Detection Level")
    plt.title("")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{model_name}/ablation/driving_perception_results.png", dpi=300)
