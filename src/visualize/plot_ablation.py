import os
import json
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":

    dirpath = f"{os.getenv('SAVE_DIR', 'results')}/topo-omni/ablation"
    file_template = "ablation_stimuli={stimuli}_localizer={localizer}_perc={percentage}_stimulate=False.json"
    # file_template = "ablation_stimuli={stimuli}_localizer={localizer}_cluster1_stimulate=False_strength=0.0.json"

    percentage = 5.0
    # file_name = f"gemini_labels_llm_as_a_judge_percentage={percentage}_cluster=largest.json"

    categories = ["faces", "bodies", "scenes", "objects"]
    
    classes = [
        "a face",
        "a body part",
        "an indoor or outdoor location",
        "a toy or object",
    ]

    # classes = [
    #     "a face",
    #     "a body part",
    #     "a location",
    #     "an object",
    # ]
    
    # percentages = [0.1, 0.5, 1, 2, 5, 10]
    # gemini_labels = read_json(os.path.join(dirpath, file_name))

    # for percentage in percentages:

    plot_data = np.ones((len(categories), len(categories))) * -1
    for localizer in categories:
        for stimuli in categories: 
            file_name = file_template.format(stimuli=stimuli, localizer=localizer, percentage=f"{percentage:.2f}")
            file_path = os.path.join(dirpath, file_name)
            if os.path.exists(file_path):
                data = read_json(file_path)
            else:
                print(f"File not found: {file_path}")
                continue
            accs = [classes[categories.index(stimuli)] in row["output"].strip().lower() for row in data["responses"]]
            # data = gemini_labels[localizer][stimuli]
            # acc = np.mean(["yes" in row["response"].strip().lower() for row in data])
            plot_data[categories.index(localizer), categories.index(stimuli)] = np.mean(accs)

    sns.set_context(context="paper", font_scale=1.5)

    sns.heatmap(plot_data, 
        annot=True, 
        fmt=".2f", 
        cmap="rocket", 
        xticklabels=categories, 
        yticklabels=categories
    )

    # plt.title(f"Accuracy After Ablating Top-{percentage}% of Clusters")
    plt.title(f"Accuracy After Ablating Largest Cluster")
    plt.ylabel("Lesion in Region Selective for")
    plt.xlabel("Stimuli")
    plt.tight_layout()
    plt.savefig(f"{dirpath}/confusion_matrix_ablation_largest_cluster_percentage={percentage}.png")
    plt.clf()
    plt.cla()
    plt.close()
