import os
import json
from glob import glob

from dotenv import load_dotenv
load_dotenv()

SAVE_DIR = os.getenv("SAVE_DIR")
CKPT_DIR = os.getenv("CKPT_DIR")

def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
def write_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

P_VALUE_THRESHOLD = 0.001
USE_AUDIO_PART = True
SMOOTH = True
FWHM = 8
TOP_K = 1

if __name__ == "__main__":

    model_name = "topo-omni"
    data_dir = f"{SAVE_DIR}/{model_name}/spacetop_clusters_figures"

    template_path = f"{data_dir}/*/island_morans_I_results_rating_audio={USE_AUDIO_PART}.json"

    results = {}
    paths = glob(template_path)
    for p in paths:
        result = read_json(p)
        cluster_id = list(result.keys())[0]
        results[cluster_id] = result[cluster_id]

    # save results
    # save_path = f"{data_dir}/island_morans_I_results_rating_contrast_vs_ratingv0_significant_p={P_VALUE_THRESHOLD}_audio={USE_AUDIO_PART}_smooth={SMOOTH}_fwhm={FWHM}.json"
    save_path = f"{data_dir}/island_morans_I_results_rating_contrast_vs_ratingv0_significant_topk={TOP_K}_audio={USE_AUDIO_PART}_smooth={SMOOTH}_fwhm={FWHM}.json"
    write_json(results, save_path)