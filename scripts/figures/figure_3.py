#!/usr/bin/env python
"""Figure 3 — visual functional organization.

Panels:
  a-d  category-selective maps + response profiles (faces / scenes / objects / word-form)
  e-f  retinotopy (polar angle + eccentricity)

precomputed: the unified in-silico selectivity sheet and the per-category response-profile
bars are regenerated from the cut's selectivity stats + response-profile pickles.
raw: additionally recompute retinotopy from the HF model over the (self-generating) stimulus
banks. See scripts/figures/_common.py for the step schema.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import run_figure  # noqa: E402

VISUAL_LOCALIZERS = ["faces", "scenes", "objects", "vwfa"]

STEPS = [
    {
        "name": "unified in-silico selectivity sheet",
        "module": "src.visualize.selectivity",
        "args": ["--results-dir", "{SAVE_DIR}/{MODEL_TITLE}", "--top-k-pct", "10"],
        "out_dir_arg": "--out-dir",
        "stage": "plot",
    },
]

# a-d response-profile bars (one plot per localizer; reads response_profiles/*.pkl from the cut)
for _loc in VISUAL_LOCALIZERS:
    STEPS.append({
        "name": f"response profile: {_loc}",
        "module": "src.visualize.plot_response_profiles",
        "args": ["--model_name", "{MODEL_TITLE}", "--localizer", _loc,
                 "--top_k_pct", "1", "--fwhm_mm", "4.0", "--anatomical_constraint", "true"],
        "collect": ["{MODEL_TITLE}/response_profiles/" + f"{_loc}_response_profiles_top1.png"],
        "stage": "plot",
    })

# e-f retinotopy (polar angle + eccentricity) — recompute-only. Needs the HF model over the
# retinotopy stimulus bank, which self-generates (public) with:
#   python -m src.utils.generate_retinotopy   # -> $STIMULI_DIR/retino_bank/{*.png, manifest.csv}
STEPS += [
    {
        "name": "retinotopy (polar angle + eccentricity)",
        "module": "src.eval.run.run_retinotopy",
        "args": ["--config", "src/configs/eval_retinotopy.yml"],
        "collect": ["{MODEL_TITLE}/retinotopy/*.png"],
        "stage": "compute",
    },
]

if __name__ == "__main__":
    sys.exit(run_figure("3", STEPS))
