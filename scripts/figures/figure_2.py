#!/usr/bin/env python
"""Figure 2c — model-guided discovery of SpaceTop clusters, validated against fMRI.

Panels:
  - per-cluster model selectivity scores (which discovered clusters are "brain-like")
  - model island-Moran's-I vs human-fMRI Moran's-I correlation across matched clusters
  - a representative discovered-cluster selectivity map on the cortical sheet

precomputed: all regenerate from the cut (cluster selectivity scores, per-cluster t-maps, and
the Moran's-I summary JSON). Running the discovery pipeline from scratch (topo-discover/) needs
the SpaceTop movies and is raw-only — see docs/DESIGN.md.

(Fig 2b — visual clusters + response profiles vs fMRI — is reproduced by figure_3.py.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import run_figure  # noqa: E402

REP_CLUSTER = "32"  # a representative brain-like (nature/landscape) cluster

STEPS = [
    {
        "name": "discovered-cluster selectivity scores",
        "script": "topo-discover/plot_model_selectivity.py",
        "args": ["--scores-path", "{SAVE_DIR}/spacetop_discovery/cluster_selectivity_scores_v1.json"],
        "out_dir_arg": "--out-dir",
        "stage": "plot",
    },
    {
        "name": "model vs fMRI Moran's-I correlation",
        "module": "src.visualize.spacetop_corr",
        "args": [],
        "collect": ["{MODEL_TITLE}/spacetop_clusters_figures/island_morans_I_results_rating_contrast_vs_ratingv0_significant_topk=1_audio=True_smooth=True_fwhm=8.png"],
        "stage": "plot",
    },
    {
        "name": f"representative cluster {REP_CLUSTER} selectivity map",
        "module": "src.visualize.spacetop_selectivity",
        "args": ["--cluster_id", REP_CLUSTER, "--topk", "1",
                 "--results-dir", "{SAVE_DIR}/{MODEL_TITLE}/spacetop_clusters_figures"],
        "collect": ["{MODEL_TITLE}/spacetop_clusters_figures/" + REP_CLUSTER + "/*.png"],
        "stage": "plot",
    },
]

if __name__ == "__main__":
    sys.exit(run_figure("2", STEPS))
