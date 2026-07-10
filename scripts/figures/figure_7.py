#!/usr/bin/env python
"""Figure 7 — model-guided discovery of novel cortical networks.

Panels:
  a  animal-selective network (cortical map + example stimuli)
  b  natural-landscape-selective network (cortical map + example stimuli)

precomputed: the cortical-sheet selectivity maps for the animal / landscape clusters, from the
per-cluster t-maps in the cut. The example-stimulus collages need the SpaceTop movie frames and
are therefore raw-only (topo-discover/make_cluster_collages.py). See docs/DESIGN.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import run_figure  # noqa: E402

# Representative discovered clusters (topo-discover merge: animals={5,6,7}, nature={30,31,32}).
NETWORKS = {"animal": "05", "landscape": "32"}

STEPS = []
for _name, _cid in NETWORKS.items():
    STEPS.append({
        "name": f"{_name} network selectivity map (cluster {_cid})",
        "module": "src.visualize.spacetop_selectivity",
        "args": ["--cluster_id", str(int(_cid)), "--topk", "1",
                 "--results-dir", "{SAVE_DIR}/{MODEL_TITLE}/spacetop_clusters_figures"],
        "collect": ["{MODEL_TITLE}/spacetop_clusters_figures/" + _cid + "/*.png"],
        "stage": "plot",
    })

# Example-stimulus collages need the SpaceTop movie frames (not redistributable) -> raw-only.
STEPS.append({
    "name": "cluster example-stimulus collages",
    "script": "topo-discover/make_cluster_collages.py",
    "args": ["--clusters_json", "topo-discover/task-alignvideo/clustering_v2/merged_clusters_tvals_v3.json",
             "--output_dir", "{OUT_DIR}/collages",
             "--video_root", "topo-discover/spacetop_embeddings_v2"],
    "stage": "compute",
    "raw_only": True,
})

if __name__ == "__main__":
    sys.exit(run_figure("7", STEPS))
