#!/usr/bin/env python
"""Compute CROSS-VALIDATED response profiles for one split half.

The cross-validation is inside the per-localizer modules, not here: they build the
selectivity mask from the half named by --odd_or_even and then measure responses on the
OTHER half's stimuli (marvi.py picks `stim_split = "odd" if trial_type == "even" else
"even"`; cognitive.py reads `<localizer>_<OTHER HALF>/`). So a full profile needs BOTH
halves run, and response_profiles/plot*.py reads the two pickles back together.

Why this driver exists — marvi.py's feature cache is keyed on (condition, stim_split) and
NOT on the localizer, because the localizer only enters through the mask. All 6 visual /
audio localizers of one half therefore share one cache. Run them in one process and the
first localizer pays for the 10 condition extractions while the other 5 are cache-only
(marvi.py loads the model lazily, so they never touch the GPU). Split them across parallel
jobs instead and they race on the same cache files. Parallelise across HALVES, which use
disjoint cache keys — see submit_topoomni_response_profiles.sh.

Usage:
    python scripts/run_response_profiles.py --family marvi     --odd_or_even odd
    python scripts/run_response_profiles.py --family cognitive --odd_or_even even
"""
import argparse
import os
import subprocess
import sys
from glob import glob

import yaml

from dotenv import load_dotenv
load_dotenv()

CKPT_DIR = os.getenv("CKPT_DIR", "./ckpts")

# Visual/audio localizers go through marvi.py; the text ones through cognitive.py. The
# default top-k differs per family, matching scripts/{marvi,cognitive}_response_profiles.sh
# and the panel filenames in the published cut (visual top1, cognitive top10).
FAMILIES = {
    "marvi": {
        "module": "src.experiments.response_profiles.marvi",
        "localizers": ["faces", "scenes", "objects", "vwfa", "bodies", "speech"],
        "top_k_pct": "1",
    },
    "cognitive": {
        "module": "src.experiments.response_profiles.cognitive",
        "localizers": ["language_text", "theory_of_mind_text", "multiple_demand_text"],
        "top_k_pct": "10",
    },
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, choices=sorted(FAMILIES))
    p.add_argument("--odd_or_even", required=True, choices=("odd", "even"),
                   help="half the selectivity mask is built from; responses are measured "
                        "on the other half")
    p.add_argument("--model_name", default="topo-omni-tiled32-4608",
                   help="run title: names both $CKPT_DIR/<run>/config.yml and the "
                        "$SAVE_DIR/<run>/ results tree")
    p.add_argument("--localizers", default=None, help="comma-separated subset")
    p.add_argument("--top_k_pct", default=None, help="default: per-family (see FAMILIES)")
    p.add_argument("--fwhm_mm", default="3.0",
                   help="smoothing FWHM, and the FWHM naming the selectivity-stats file to "
                        "read. Must match the sweep that produced the stats.")
    p.add_argument("--anatomical_constraint", default="true", choices=("true", "false"))
    p.add_argument("--significant_q", type=float, default=None,
                   help="keep only top-k units that are ALSO FDR-significant at q < this "
                        "value with t > 0. 0.05 matches the alpha the selectivity sweep "
                        "was run at. Omit for a pure top-k mask.")
    p.add_argument("--coords", default=None,
                   help="coords.npy the run was trained with. Default: read position-dir "
                        "out of the run's config.yml, which is what selectivity/run.py does.")
    p.add_argument("--checkpoint", default=None,
                   help="default: highest-numbered checkpoint-* in the run dir")
    args = p.parse_args()

    fam = FAMILIES[args.family]
    localizers = ([l.strip() for l in args.localizers.split(",") if l.strip()]
                  if args.localizers else list(fam["localizers"]))
    unknown = [l for l in localizers if l not in fam["localizers"]]
    if unknown:
        raise SystemExit(f"unknown {args.family} localizer(s) {unknown}; "
                         f"known: {fam['localizers']}")
    top_k_pct = args.top_k_pct or fam["top_k_pct"]

    run_dir = os.path.join(CKPT_DIR, args.model_name)
    run_config = {}
    cfg_path = os.path.join(run_dir, "config.yml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            run_config = yaml.safe_load(f) or {}

    env = os.environ.copy()
    # Resolved at IMPORT time in spatial_utils and it fixes the sheet dimensions the mask
    # and the smoother are built against. Evaluating a tiled32 run under the default strip
    # layout samples the wrong units, so take it from the config the run was trained with
    # rather than trusting the caller -- same reasoning as scripts/run_selectivity.py.
    layout = run_config.get("sheet-layout")
    if layout:
        env["TOPO_SHEET_LAYOUT"] = layout
        print(f"> TOPO_SHEET_LAYOUT={layout} (from {args.model_name}/config.yml)")

    coords = args.coords
    if not coords:
        pos_dir = (run_config.get("topo-params") or {}).get("position-dir")
        if pos_dir:
            for cand in (os.path.join("src", pos_dir, "coords.npy"),
                         os.path.join(pos_dir, "coords.npy")):
                if os.path.exists(cand):
                    coords = cand
                    break
    if coords:
        print(f"> coords: {coords}")
    else:
        print("> WARNING: no coords.npy resolved; the modules will assume the identity "
              "grid, which is wrong for any run whose units were shuffled.")

    ckpt = args.checkpoint
    if not ckpt:
        found = sorted(glob(f"{run_dir}/checkpoint-*"),
                       key=lambda q: int(q.rsplit("-", 1)[1]))
        ckpt = found[-1] if found else (run_dir if os.path.exists(run_dir) else None)
    if ckpt:
        print(f"> checkpoint: {ckpt}")

    print(f"> {args.family}: {len(localizers)} localizer(s), {args.odd_or_even} half, "
          f"top{top_k_pct}, fwhm {args.fwhm_mm}")

    failures = []
    for loc in localizers:
        cmd = [sys.executable, "-m", fam["module"],
               "--model_name", args.model_name,
               "--localizer", loc,
               "--odd_or_even", args.odd_or_even,
               "--top_k_pct", str(top_k_pct),
               "--fwhm_mm", str(args.fwhm_mm)]
        if args.significant_q is not None:
            cmd += ["--significant_q", str(args.significant_q)]
        if args.family == "marvi":
            cmd += ["--anatomical_constraint", args.anatomical_constraint]
        else:
            # cognitive.py names the stats file separately from the smoother FWHM; keep the
            # two equal so the mask and the responses live in the same smoothed space.
            cmd += ["--stats_fwhm_mm", str(args.fwhm_mm)]
        if coords:
            cmd += ["--coords", coords]
        if ckpt:
            cmd += ["--checkpoint", ckpt]

        print(f"\n=== {args.family}/{loc} [{args.odd_or_even}] ===\n    {' '.join(cmd)}")
        if subprocess.run(cmd, env=env).returncode != 0:
            failures.append(loc)
            print(f"    x {loc} failed")

    if failures:
        raise SystemExit(f"FAILED localizer(s): {', '.join(failures)}")
    print(f"\nAll {len(localizers)} {args.family} localizer(s) done "
          f"({args.odd_or_even} half).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
