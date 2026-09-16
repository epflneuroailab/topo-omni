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

# Objects was #EFC94C, dE2000 = 3.0 from the Scenes gold beside it. Matches
# selectivity/plot.py and the response-profile plotters.
color_palette = sns.color_palette(["#E64B35", "#F39B2F", "#F1C232", "#A0522D"])

if __name__ == "__main__":

    import argparse
    _p = argparse.ArgumentParser(description=__doc__)
    _p.add_argument("--model_name", default="topo-omni",
                    help="run title under $SAVE_DIR holding ablation/ (was hardcoded).")
    _p.add_argument("--percentages", default="5,10,15,20,25,30",
                    help="coverage levels on the x axis; each needs its own judged file.")
    _p.add_argument("--width", type=float, default=7.0,
                    help="figure width in inches (was a hardcoded 4.0, which read as very "
                         "compact for these panels).")
    _p.add_argument("--height", type=float, default=4.0, help="figure height in inches.")
    _args = _p.parse_args()

    model_name = _args.model_name
    dirpath = f"{SAVE_DIR}/{model_name}/ablation"
    # _v3 -> _v4: similarity.py writes "_v4", so the _v3 template matched nothing it has
    # ever produced and every percentage was skipped.
    _str = ""   # see run.py: stimulation has no strength parameter
    path_template = ("similarity_ablation_results_top{percentage}_stimulate=True"
                     + _str + "_v4.json")

    percentages = [int(x) for x in _args.percentages.split(",") if x.strip()]

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

    plt.figure(figsize=(_args.width, _args.height))

    # hue_order + a dict palette, because a bare list palette is assigned in the order the
    # hue levels first APPEAR in the frame. similarity.py skips the matched diagonal, so
    # localizers first appear as bodies, scenes, objects, faces -- and faces was being
    # drawn in the Objects colour, scenes in the Faces colour, and so on. The legend is
    # suppressed below, so nothing in the figure revealed the swap.
    hue_order = ["Faces", "Bodies", "Scenes", "Objects"]
    cat_colors = {"Faces": "#E64B35", "Bodies": "#F39B2F",
                  "Scenes": "#F1C232", "Objects": "#A0522D"}

    sns.lineplot(
        data=results,
        x="percentage",
        y="perceived",
        hue="localizer",
        hue_order=hue_order,
        marker="o",
        dashes=False,
        markersize=15,
        palette=cat_colors,
    )

    sns.despine()

    # remove legend
    plt.legend([], frameon=False)

    plt.xlabel("Coverage (%)")
    plt.ylabel("Detection Level")
    plt.title("")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{model_name}/ablation/driving_perception_results.png", dpi=300)
