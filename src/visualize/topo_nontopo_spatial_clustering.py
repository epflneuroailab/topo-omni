import os
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

SAVE_DIR = os.getenv("SAVE_DIR", "./results")

def read_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":

    topo_model_name = "qwen2_5_3b_spatial_task_final_7"
    non_topo_model_name = "qwen2_5_3b_task_7"

    localizers = [
        # ("faces", "vision", "Faces"), 
        # ("bodies", "vision", "Bodies"), 
        # ("scenes", "vision", "Scenes"), 
        # ("objects", "vision", "Objects"), 
        # ("vwfa", "vision", "Words"), 
        # ("speech", "audio", "Quilted Speech"), 
        # ("vocals", "audio", "False Photo"), 
        ("language_text_ALL", "language", "Nonwords"),
        ("multiple_demand_text_ALL", "cognitive", "Math"),
        # ("theory_of_mind_text_ALL", "cognitive", "False Belief"),
    ]

    for category, modality, name in localizers:

        topo_filepath = f"{SAVE_DIR}/{topo_model_name}/{category}/{category}_fwhm4.0_island_morans_I.json"
        non_topo_filepath = f"{SAVE_DIR}/{non_topo_model_name}/{category}/{category}_fwhm4.0_island_morans_I.json"

        topo_stats = read_json(topo_filepath)
        non_topo_stats = read_json(non_topo_filepath)

        topo_morans_I = [vals["moran_I"] for vals in topo_stats.values()]
        non_topo_morans_I = [vals["moran_I"] for vals in non_topo_stats.values()]

        df = pd.DataFrame({
            "Moran's I": topo_morans_I + non_topo_morans_I,
            "Model": ["Topo-Omni"] * len(topo_morans_I) + ["Non-Topo"] * len(non_topo_morans_I)
        })

        sns.set_theme(style="whitegrid", font_scale=2)

        fig, ax = plt.subplots(figsize=(4, 5))
        # plot a barplot with error bars showing mean and standard deviation of Moran's I for topo and non-topo models
        sns.barplot(data=df,
            x="Model",
            y="Moran's I",
            errorbar="se",
            palette="Blues_r",
            legend=False,
            width=0.8,
            ax=ax,
        )
        sns.despine()
        plt.xlabel("")
        plt.xticks([])
        plt.ylabel("Island Moran's I")
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/comparison/{category}_spatial_clustering.png", dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()
        plt.cla()

