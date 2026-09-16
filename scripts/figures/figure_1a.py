#!/usr/bin/env python
"""Figure 1a — the unified in-silico cortical sheet (tiled32 model).

One panel: every category localizer overlaid on the same 800x576 tiled32 sheet, reduced to
the three largest selective islands per category (--n-islands 3), taken from the top-10%%
of units that survive significance testing at p<0.05.

Model: the ``topo-omni-tiled32-4608`` run (checkpoint-730), NOT the default ``topo-omni``
strip model the other figures use — the tiled32 sheet is 800x576 (460,800 units) and its
own ``config.yml`` carries ``sheet-layout: tiled32``, which ``scripts/run_selectivity.py``
lifts into ``$TOPO_SHEET_LAYOUT`` before the model is built. Evaluating this run under the
default strip layout mismatches the coords and scrambles every map.

precomputed: draws from <SAVE_DIR>/topo-omni-tiled32-4608/<category>/<category>_fwhm3.0_
             selectivity_stats.pkl — the smoothed (FWHM 3 mm) selectivity sweep.
raw:         first re-extracts those stats from the checkpoint over the localizer stimuli
             (GPU + $STIMULI_DIR), then draws.

Two flags beyond the plotting selection itself, both load-bearing:
  --stats-suffix _fwhm3.0  the smoothed sweep writes '<cat>_fwhm3.0_...pkl'; plot.py's
                           default guess looks for '<cat>_all_...pkl' / '<cat>_...pkl' and
                           would either miss the file or silently pick up a stale
                           unsmoothed sweep from an older run.
  --pre-gridded            run_selectivity smooths BEFORE the t-test, so stats['t'] is
                           already in grid raster order. Placing it by reshape is correct;
                           scattering it through coords.npy would permute it a second time
                           (nearest-neighbour r falls +0.983 -> +0.002 on tiled32 faces).
"""
import os
import sys
from pathlib import Path

# MUST be set before the plot step starts: src/models/spatial_utils.py resolves
# DEFAULT_SHEET_LAYOUT from $TOPO_SHEET_LAYOUT at IMPORT time, and plot.py reads its sheet
# dimensions from it. run_selectivity.py lifts this out of the run's config.yml for the
# *eval* step, but the plot step runs on its own, so without this it renders the tiled32
# stats (460,800 units) against the default strip sheet (309x512) and dies with
# "cannot reshape array of size 460800 into shape (512,309)". _common.run_figure passes
# os.environ through to each subprocess, so setting it here is enough.
os.environ["TOPO_SHEET_LAYOUT"] = "tiled32"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import run_figure  # noqa: E402

# The tiled32 run. _common.MODEL_TITLE ("topo-omni") is the strip model used by figures
# 2-7, so this figure names its run explicitly instead of using the {MODEL_TITLE} token.
RUN_TITLE = "topo-omni-tiled32-4608"
FWHM = "3.0"

STEPS = [
    # ---- raw only: re-extract the selectivity stats from the checkpoint -----------------
    # Spawns one `src.experiments.selectivity.run` per localizer (10 of them: 6 MARVi video,
    # 3 text, 1 Pernet audio). Reads the checkpoint from $CKPT_DIR/<RUN_TITLE> (highest
    # checkpoint-* wins) and the stimuli from $STIMULI_DIR.
    {
        "name": "selectivity stats (all localizers, FWHM 3mm)",
        "script": "scripts/run_selectivity.py",
        "args": [
            "--model_name", RUN_TITLE,
            "--config_path", "src/configs/eval/marvi.yml",
            "--fwhm", FWHM,
        ],
        "stage": "compute",
    },
    # ---- the panel ----------------------------------------------------------------------
    {
        "name": "unified selectivity map (3 largest islands per category)",
        "module": "src.experiments.selectivity.plot",
        "args": [
            "--results-dir", "{SAVE_DIR}/" + RUN_TITLE,
            "--stats-suffix", f"_fwhm{FWHM}",
            "--pre-gridded",
            "--morph-clean",
            "--n-islands", "3",
            "--p-threshold", "0.05",
            "--top-k-pct", "10",
            "--no-svg",
        ],
        "out_dir_arg": "--out-dir",
        "stage": "plot",
    },
]

if __name__ == "__main__":
    sys.exit(run_figure("1a", STEPS))
