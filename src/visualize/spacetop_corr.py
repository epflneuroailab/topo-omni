import os
import json
import argparse
from pathlib import Path

import matplotlib
import seaborn as sns

import numpy as np
from scipy.stats import pearsonr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

SAVE_DIR = os.getenv("SAVE_DIR")
CKPT_DIR = os.getenv("CKPT_DIR")

P_VALUE_THRESHOLD = 0.001
USE_AUDIO_PART = True
SMOOTHING = True
FWHM = 8
TOP_K = 1

fMRI_I_sig = {
    2: 0.4597612354840860,
    3: 0.5523143151796300,
    14: 0.5633570631106290,
    25: 0.5883768800618310,
    31: 0.4543576809372930,
    32: 0.4776345218511720,
    34: 0.8488107077833800,
    47: 0.4728499203501310,
    48: 0.4943987536463650,
    49: 0.5669101546914130,
}

weighted_fMRI_I = {
    31: 0.8570475577364020,
    2: 0.7943399649322180,
    47:0.8921872379059740,
    32:0.8604386655674470,
    48:0.8912912796660160,
    3: 0.846015980801176,
    14:0.8655084269031190,
    49:0.8587795759118490,
    25:0.7653908266014580,
    34:0.9081691677210860
}

weighted_fMRI_vs_rating_sig = {
    45:  0.4651736806304290,
    38:  0.4909482445828140,
    23:  0.5932826114739280,
    30:  0.6341886049033240,
    43:  0.6363279337572190,
    12:  0.6372922021445840,
    24:  0.6393056064008830,
    46:  0.6497590207604390,
    21:  0.6901044423866310,
    10:  0.6908963766725660,
    11:  0.6912335165270980,
    25:  0.7162266170532090,
    8:  0.7289490338767300,
    37:  0.7333689110561990,
    32:  0.7429455416514210,
    28:  0.7681474624658720,
    2:  0.7703316813362680,
    3:  0.7743211668183500,
    19:  0.7846731407524390,
    40:  0.7851815318126720,
    31:  0.7949205601032850,
    22:  0.8011311480495270,
    27:  0.8049559453161410,
    39:  0.8232892038915880,
    7:  0.8275667756633430,
    49:  0.8279178172875030,
    14:  0.828119726672421,
    18:  0.858794832562459,
    33:  0.8647174024483970,
    42:  0.8738570927666720,
    13:  0.8762960487987950,
    26:  0.884565769306839,
    48:  0.8858407572813760,
    41:  0.887049221187876,
    44:  0.8898449320457720,
    34:  0.917993366264998
}

fMRI_vs_rating_sig = {
    45:  0.408160054043148,
    48:  0.4563253109645700,
    24:  0.4713862030520800,
    23:  0.48362538535050900,
    38:  0.4909482445828140,
    49:  0.5063406959479930,
    8:  0.510267909903613,
    2:  0.5109712595400380,
    30:  0.5456864966144920,
    21:  0.5485112713723760,
    26:  0.5570042605009470,
    33:  0.5579355850479270,
    32:  0.5589226581585330,
    14:  0.5592564935213700,
    10:  0.5617803533508300,
    11:  0.5652623219862290,
    3:  0.5731070702897130,
    43:  0.5745873424090550,
    12:  0.5759403189780930,
    44:  0.5821554304351580,
    37:  0.589549437786702,
    31:  0.6123896919384660,
    41:  0.6124936442536170,
    13:  0.6135786260151160,
    46:  0.616335628375828,
    39:  0.6249914855076120,
    19:  0.6387611452699470,
    22:  0.6518537462257000,
    28:  0.6563631344155060,
    42:  0.6599283415885790,
    25:  0.6643866852848930,
    18:  0.7089152273079470,
    7:  0.7292258729103960,
    40:  0.7343317839475170,
    27:  0.7412689426392820,
    34:  0.8306888504323200,
}


fMRI_vs_rating_all = {
    47: 0.1478192683916650,
    49: 0.1644567310959120,
    48: 0.2016799383259340,
    44: 0.21275879489211100,
    41: 0.2336399889605660,
    34: 0.2489348761585130,
    8: 0.2930207161158450,
    2: 0.29650257852333900,
    45: 0.30416418394280800,
    24: 0.3129356732291720,
    42: 0.31631497193442500,
    33: 0.3258143637520380,
    39: 0.3386637011114100,
    30: 0.3478909624623960,
    26: 0.3591481846754440,
    3: 0.3653644268980410,
    11: 0.3700959148768710,
    23: 0.38284876303683800,
    10: 0.3837845487039440,
    21: 0.3882469274049440,
    19: 0.3990008686398580,
    28: 0.40028115908680200,
    14: 0.4008573994231870,
    37: 0.4234689581169420,
    12: 0.4249999731964060,
    22: 0.4316218747449210,
    46: 0.4371437682307280,
    13: 0.43786316301885400,
    32: 0.4449527691555400,
    31: 0.47792448039437500,
    18: 0.4791484361349540,
    38: 0.4909482445828140,
    43: 0.5140568352327220,
    25: 0.5149587349837730,
    40: 0.54416498096242,
    27: 0.5655851363095320,
    7: 0.6063226894551500,
}

weighted_fMRI_vs_rating_all = {
    47:  0.1478192683916650,
    45:  0.42083026126569500,
    38:  0.4909482445828140,
    23:  0.5661069587505690,
    30:  0.5753173477428360,
    24:  0.5862887663335050,
    12:  0.593340120277285,
    43:  0.6091166552398370,
    46:  0.6174946185557060,
    8:  0.6203079775455770,
    10:  0.6486359165814750,
    21:  0.6594941793253780,
    11:  0.6600128460243780,
    37:  0.6757200885167510,
    25:  0.6920999304126340,
    32:  0.7315726651156790,
    2:  0.7420314090822390,
    19:  0.7470973648566090,
    28:  0.7495054946386200,
    3:  0.7521664293305580,
    40:  0.7616321225871560,
    31:  0.7857267615434570,
    22:  0.7889339659477880,
    27:  0.7909234162611140,
    39:  0.806015110151026,
    49:  0.8061067267696520,
    14:  0.8189612844147690,
    7:  0.8209335124012160,
    33:  0.8415336058482390,
    18:  0.8452365221164080,
    42:  0.8565274934122930,
    44:  0.8671208233188910,
    13:  0.8691333963347530,
    41:  0.8702787581600050,
    48:  0.8712807334167100,
    26:  0.8744801675509770,
    34:  0.9164016421216190,
}

fMRI_vs_rating_standard_morans_I = {
   41: 0.873556201308277,
   44: 0.874322293081482,
   18: 0.877779354716203,
   5: 0.878843222023851,
   4: 0.881851910266635,
   24: 0.887628678447746,
   10: 0.894492128027268,
   43: 0.89458826558618,
   23: 0.896171037449613,
   28: 0.899911996097823,
   11: 0.906601538496302,
   49: 0.907949867805455,
   45: 0.913151657110638,
   29: 0.916311521547417,
   15: 0.916363262291931,
   6: 0.917079473990804,
   9: 0.922834422930607,
   30: 0.922896555153743,
   16: 0.924852936517173,
   1: 0.927761974353503,
   0: 0.928339817957869,
   14: 0.930495476477469,
   34: 0.931865849641441,
   8: 0.933123627403571,
   13: 0.935479328864871,
   12: 0.936484626595227,
   46: 0.93666172315938,
   3: 0.938220799067045,
   17: 0.940638288289132,
   48: 0.941943622123283,
   2: 0.942104110445276,
   22: 0.942767442552222,
   21: 0.942947017180937,
   32: 0.944048704559673,
   31: 0.946227117096355,
   26: 0.946380848882852,
   35: 0.946963025714602,
   47: 0.950332428421175,
   19: 0.950417180499124,
   39: 0.950526019216783,
   42: 0.950640989662555,
   33: 0.951407398803956,
   7: 0.954058375383104,
   27: 0.954411013400934,
   40: 0.956865736169559,
   38: 0.96055895348786,
   37: 0.970105081367294,
   36: 0.970927026791653,
   20: 0.971240770940712,
   25: 0.972053558700672,
}

def read_json(path):
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)

def cluster_key_to_id(cluster_key):
    return int(cluster_key.split("_")[-1])

def build_matched_pairs(island_data, fmri_map):
    matched_rows = []
    for cluster_key, values in island_data.items():
        cluster_id = cluster_key_to_id(cluster_key)
        if cluster_id in fmri_map:
            # if values["total_num_islands"] > 1:
            matched_rows.append(
                {
                    "cluster_id": cluster_id,
                    "I_value": float(values["I"]),
                    "fMRI_I": float(fmri_map[cluster_id]),
                }
            )
    return sorted(matched_rows, key=lambda x: x["cluster_id"])


def plot_correlation(matched_rows, output_path):
    x = np.array([row["I_value"] for row in matched_rows], dtype=float)
    y = np.array([row["fMRI_I"] for row in matched_rows], dtype=float)
    r, p = pearsonr(x, y)
    r = float(r)
    p = float(p)

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, color="tab:blue", alpha=0.9)

    m, b = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_fit, m * x_fit + b, color="tab:red", linestyle="--", linewidth=1.5)

    for row in matched_rows:
        plt.annotate(str(row["cluster_id"]), (row["I_value"], row["fMRI_I"]), fontsize=8, alpha=0.85)

    sns.despine()
    plt.xlabel("Model | Island Moran's I")
    plt.ylabel("Brain | Island Moran's I")
    plt.title(f"Model vs fMRI (n={len(matched_rows)}, Pearson r={r:.3f}, p={p:.4g}) | FWHM={FWHM}mm, Top-{TOP_K}%")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return r, p


def main():
    parser = argparse.ArgumentParser(
        description="Plot correlation between island Moran's I and fMRI_I for matching cluster IDs."
    )
    parser.add_argument(
        "--input-json",
        default=f"island_morans_I_results_rating_contrast_vs_ratingv0_significant_topk={TOP_K}_audio={USE_AUDIO_PART}_smooth={SMOOTHING}_fwhm={FWHM}.json",
        # default=f"island_morans_I_results_rating_contrast_vs_ratingv0_significant_p={P_VALUE_THRESHOLD}_audio={USE_AUDIO_PART}_smooth={SMOOTHING}_fwhm={FWHM}.json",
        help="Path to JSON file containing cluster data with key format cluster_XX and field I-value.",
    )
    parser.add_argument(
        "--output-path",
        default=f"island_morans_I_results_rating_contrast_vs_ratingv0_significant_topk={TOP_K}_audio={USE_AUDIO_PART}_smooth={SMOOTHING}_fwhm={FWHM}.png",
        # default=f"island_morans_I_results_rating_contrast_vs_ratingv0_significant_p={P_VALUE_THRESHOLD}_audio={USE_AUDIO_PART}_smooth={SMOOTHING}_fwhm={FWHM}.png",
        help="Path to save correlation plot.",
    )
    args = parser.parse_args()

    model_name = "qwen2_5_3b_spatial_task_final_7"
    data_dir = f"{SAVE_DIR}/{model_name}/spacetop_clusters_figures"
    input_json = os.path.join(data_dir, args.input_json)
    output_path = os.path.join(data_dir, args.output_path)
    
    island_data = read_json(input_json)
    matched_rows = build_matched_pairs(island_data, fMRI_vs_rating_all)
    if not matched_rows:
        raise ValueError("No overlapping cluster IDs found between input JSON and fMRI_I mapping.")

    r, p = plot_correlation(matched_rows, output_path)
    print(f"Matched clusters: {len(matched_rows)}")
    print("Cluster IDs:", [row["cluster_id"] for row in matched_rows])
    print(f"Pearson correlation r: {r:.4f}")
    print(f"Pearson correlation p-value: {p:.6g}")
    print(f"Saved plot to: {Path(output_path).resolve()}")


if __name__ == "__main__":
    main()
