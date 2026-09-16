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
    "objects": "#A0522D",
}

# Matches selectivity/plot.py and response_profiles/*. Objects was #EFC94C, dE2000 = 3.0
# from the Scenes gold beside it -- the two bars read as one colour.
color_palette = sns.color_palette(["#E64B35", "#F39B2F", "#F1C232", "#A0522D"])

if __name__ == "__main__":

    import argparse
    _p = argparse.ArgumentParser(description=__doc__)
    _p.add_argument("--model_name", default="topo-omni",
                    help="run title under $SAVE_DIR holding ablation/ (was hardcoded).")
    _p.add_argument("--percentage", type=float, default=10.0)
    _p.add_argument("--width", type=float, default=7.0,
                    help="figure width in inches (was a hardcoded 4.0, which read as very "
                         "compact for these panels).")
    _p.add_argument("--height", type=float, default=4.0, help="figure height in inches.")
    _p.add_argument("--ymax", type=float, default=None,
                    help="pin the y-axis top (default: auto from the data). Use it to make "
                         "the two panels share a scale.")
    _args = _p.parse_args()

    dirpath = f"{os.getenv('SAVE_DIR', 'results')}/{_args.model_name}/ablation"
    file_template = "similarity_ablation_results_top{percentage}_stimulate=False_v4.json"

    no_ablation_file = "similarity_no_ablation_results_v4.json"
    no_ablation_data = read_json(os.path.join(dirpath, no_ablation_file)) 

    percentage = _args.percentage

    categories = ["faces", "bodies", "scenes", "objects"]
    
    classes = [
        "a face",
        "a body part",
        "an indoor or outdoor location",
        "a toy or object",
    ]

    model_name = _args.model_name
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

    fig, ax = plt.subplots(figsize=(_args.width, _args.height))
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
    # ylim was hardcoded to (0, 0.9). On this run two of the three bars sit ABOVE it --
    # suppressing bodies leaves face accuracy at 0.985 and objects at 0.939 -- so both bars
    # were drawn clipped at the top edge with their error bars entirely off-plot, which
    # reads as "those bars have no error bars". Anchor the baseline at 0 and let the top
    # follow the data: matplotlib has already autoscaled it to include the bars, the error
    # bars and the no-ablation line. --ymax pins it when panels need to match.
    ax.set_ylim(0, _args.ymax if _args.ymax else ax.get_ylim()[1])
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{model_name}/ablation/suppressing_face_identification_results.png", dpi=300)
    plt.clf()
    plt.cla()
    plt.close()


    fig, ax = plt.subplots(figsize=(_args.width, _args.height))

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
    # ylim was hardcoded to (0, 0.9). On this run two of the three bars sit ABOVE it --
    # suppressing bodies leaves face accuracy at 0.985 and objects at 0.939 -- so both bars
    # were drawn clipped at the top edge with their error bars entirely off-plot, which
    # reads as "those bars have no error bars". Anchor the baseline at 0 and let the top
    # follow the data: matplotlib has already autoscaled it to include the bars, the error
    # bars and the no-ablation line. --ymax pins it when panels need to match.
    ax.set_ylim(0, _args.ymax if _args.ymax else ax.get_ylim()[1])
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{model_name}/ablation/suppressing_face_region_results.png", dpi=300)
    plt.clf()
    plt.cla()
    plt.close()


