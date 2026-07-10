import os
import json
import pandas as pd
from glob import glob

from dotenv import load_dotenv
load_dotenv()

SAVE_DIR = os.getenv("SAVE_DIR")

def write_json(data, file):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":

    group_ids = ["G1", "G2", "G3"]

    dirpath = f"{SAVE_DIR}/topo-omni/spacetop_clusters_iou"
    results = {}
    for group_id in group_ids:
        files = glob(os.path.join(dirpath, f"{group_id}_*", "iou_summary_*.csv"))

        for file_path in files:
            data = pd.read_csv(file_path, header=0)
            cluster_ids = data["cluster_ids"].item()
            observed_iou = data["observed_iou"].item()
            p_value = data["p_value"].item()
            results[cluster_ids] = {
                "observed_iou": observed_iou,
                "p_value": p_value
            }

    results = dict(sorted(results.items(), key=lambda x: x[1]["observed_iou"], reverse=True))
    write_json(results, os.path.join(dirpath, f"all_iou_summary.json"))