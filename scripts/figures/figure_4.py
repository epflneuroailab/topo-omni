#!/usr/bin/env python
"""Figure 4 — auditory functional organization.

Panels:
  a  speech localizer (response profile)
  b  voice / PerNet localizer (vocal vs non-vocal response profile)
  c  tonotopy (preferred-frequency map)

precomputed: 4a/4b response-profile bars from the cut's response-profile pickles.
raw: 4c tonotopy additionally recomputed from the HF model over the pure-tone bank
(regenerate the bank with `python -m src.experiments.tonotopy.stimuli`).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import run_figure  # noqa: E402

STEPS = [
    {
        "name": "response profile: speech",
        "module": "src.experiments.response_profiles.plot",
        "args": ["--model_name", "{MODEL_TITLE}", "--localizer", "speech",
                 "--top_k_pct", "1", "--fwhm_mm", "4.0", "--anatomical_constraint", "true"],
        "collect": ["{MODEL_TITLE}/response_profiles/speech_response_profiles_top1.png"],
        "stage": "plot",
    },
    {
        "name": "response profile: voice / PerNet",
        "module": "src.experiments.response_profiles.plot_pernet",
        "args": ["--model_name", "{MODEL_TITLE}", "--localizer", "pernet",
                 "--top_k_pct", "1", "--fwhm_mm", "4.0", "--anatomical_constraint", "true"],
        "collect": ["{MODEL_TITLE}/response_profiles/pernet_response_profiles_top1.png"],
        "stage": "plot",
    },
    {
        "name": "tonotopy map",
        "module": "src.experiments.tonotopy.run",
        "args": ["--config", "src/configs/eval/tonotopy.yml"],
        "collect": ["{MODEL_TITLE}/tonotopy/tonotopy_selectivity_*.png"],
        "stage": "compute",
    },
]

if __name__ == "__main__":
    sys.exit(run_figure("4", STEPS))
