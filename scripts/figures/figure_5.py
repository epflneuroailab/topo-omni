#!/usr/bin/env python
"""Figure 5 — higher cognitive networks (response profiles).

Panels:
  a  language-selective network (sentences vs non-words)
  b  multiple-demand network
  c  theory-of-mind network

precomputed: response-profile bars for each cognitive localizer, from the cut's
response-profile pickles (top-10%). See scripts/figures/_common.py for the step schema.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import run_figure  # noqa: E402

COGNITIVE_LOCALIZERS = ["language_text", "multiple_demand_text", "theory_of_mind_text"]

STEPS = [
    {
        "name": f"response profile: {loc}",
        "module": "src.visualize.plot_cog_response_profiles",
        "args": ["--localizer", loc, "--top_k_pct", "10"],
        "collect": ["{MODEL_TITLE}/response_profiles/" + f"{loc}_response_profiles_top10.png"],
        "stage": "plot",
    }
    for loc in COGNITIVE_LOCALIZERS
]

if __name__ == "__main__":
    sys.exit(run_figure("5", STEPS))
