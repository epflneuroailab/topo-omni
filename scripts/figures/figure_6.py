#!/usr/bin/env python
"""Figure 6 — causal control of visual perception.

Panels:
  a    driving perception by stimulating face-selective units (detection vs coverage)
  b-c  categorization accuracy after suppressing each region / stimulus type

precomputed: all panels regenerate from the ablation/stimulation result JSONs in the cut
(the LLM-judge accuracy JSONs). raw regeneration of those JSONs needs the HF model and the
localizer stimuli (not redistributable) — see docs/DESIGN.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import run_figure  # noqa: E402

STEPS = [
    {
        "name": "driving perception (stimulation)",
        "module": "src.visualize.driving_perception",
        "args": [],
        "collect": ["{MODEL_TITLE}/ablation/driving_perception_results.png"],
        "stage": "plot",
    },
    {
        "name": "suppression accuracy bars",
        "module": "src.visualize.plot_ablation_barplot",
        "args": [],
        "collect": ["{MODEL_TITLE}/ablation/suppressing_face_identification_results.png",
                    "{MODEL_TITLE}/ablation/suppressing_face_region_results.png"],
        "stage": "plot",
    },
]

if __name__ == "__main__":
    sys.exit(run_figure("6", STEPS))
