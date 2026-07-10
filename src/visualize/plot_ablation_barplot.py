import os
import json
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

SAVE_DIR = os.getenv("SAVE_DIR")

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

category_to_color = {
    "faces": "#E64B35",
    "bodies": "#F39B2F",
    "scenes": "#F1C232",
    "objects": "#EFC94C",
}

color_palette = sns.color_palette(["#E64B35", "#F39B2F", "#F1C232", "#EFC94C"])

if __name__ == "__main__":

    dirpath = f"{os.getenv('SAVE_DIR', 'results')}/topo-omni/ablation"
    file_template = "similarity_ablation_results_top{percentage}_stimulate=False_v4.json"

    no_ablation_file = "similarity_no_ablation_results_v4.json"
    no_ablation_data = read_json(os.path.join(dirpath, no_ablation_file)) 

    percentage = 10.0

    categories = ["faces", "bodies", "scenes", "objects"]
    
    classes = [
        "a face",
        "a body part",
        "an indoor or outdoor location",
        "a toy or object",
    ]

    model_name = "topo-omni"
    plot_data = []

    file_name = file_template.format(percentage=f"{percentage:.0f}")
    file_path = os.path.join(dirpath, file_name)
    data = read_json(file_path)

    for sample in data:
        plot_data += [{
            "localizer": sample["localizer"],
            "stimuli": sample["stimuli"],
            "accuracy": "yes" in sample["generated_text"].strip().lower(),
        }]

    for sample in no_ablation_data:
        plot_data += [{
            "localizer": "No ablation",
            "stimuli": sample["stimuli"],
            "accuracy": sample["valid"],
        }]

    sns.set_theme(style="whitegrid", context="paper", font_scale=2)


    df = pd.DataFrame(plot_data)
    df["stimuli"] = df["stimuli"].str.capitalize()
    df["localizer"] = df["localizer"].str.capitalize()

    df_stim_faces = df[df["stimuli"] == "Faces"]
    df_localizer_faces = df[df["localizer"] == "Faces"]

    fig, ax = plt.subplots(figsize=(4, 4))  # narrower figure pushes bars together
    sns.barplot(
        data=df_stim_faces[df_stim_faces["localizer"] != "No ablation"],
        x="localizer",
        y="accuracy",
        palette=color_palette,
        width=0.6,
        ax=ax,
    )

    sns.despine()

    # Get no-ablation accuracy for faces stimuli
    no_ablation_face_acc = df_stim_faces[df_stim_faces["localizer"] == "No ablation"]["accuracy"].mean()

    # Draw a dotted line spanning the full plot
    ax.axhline(y=no_ablation_face_acc, color="black", linestyle="--", linewidth=1.5, label="No Ablation")
    # ax.legend(fontsize=12)

    # plt.title(f"Accuracy After Ablating Top-{percentage}% of Clusters")
    # plt.title(f"Suppressing Top-{int(percentage)}% of Clusters")


    plt.xticks(fontsize=12)
    plt.ylabel("Face Accuracy")
    plt.xlabel("Region to Suppress")
    plt.ylim(0, 0.9)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{model_name}/ablation/suppressing_face_identification_results.png", dpi=300)
    plt.clf()
    plt.cla()
    plt.close()


    fig, ax = plt.subplots(figsize=(4, 4))  # narrower figure pushes bars together

    sns.barplot(
        data=df_localizer_faces[df_localizer_faces["localizer"] != "No ablation"],
        x="stimuli",
        y="accuracy",
        palette=color_palette,
        width=0.6,
        ax=ax,
    )

    sns.despine()
    no_ablation_df = df[df["localizer"] == "No ablation"]
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]

    for i, cat in enumerate(tick_labels):
        baseline = no_ablation_df[no_ablation_df["stimuli"] == cat]["accuracy"].mean()
        bar_width = 0.5
        half = bar_width / 2
        ax.plot(
            [i - half, i + half],  # short horizontal segment centered on the bar
            [baseline, baseline],
            color="black", linestyle="--", linewidth=1.5,
            label="No Ablation" if i == 0 else None,
        )

    # plt.title(f"Accuracy After Ablating Top-{percentage}% of Clusters")
    # plt.title(f"Suppressing Top-{int(percentage)}% of Face Clusters")
    plt.xticks(fontsize=12)
    plt.ylabel("Stimuli Accuracy")
    plt.xlabel("Stimuli")
    plt.ylim(0, 0.9)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{model_name}/ablation/suppressing_face_region_results.png", dpi=300)
    plt.clf()
    plt.cla()
    plt.close()


