#!/usr/bin/env python
"""Pernet 2015 — Stage 1 driver: precomputed cut -> paper figures.

Figure lineages (allow-list from the code-to-figure index; docs/DESIGN.md §2.4):
  - Fig. 3b surface map      : 00 GLM -> 01 group -> 02 surface           [PORTED]
  - Fig. 3b 2-bar profile    : cv_04 group fROI -> cv_05 responses+plot   [PORTED]
  - Fig. B3b island Moran's I: 05 (core.spatial_stats) -> 06 3-bar comparison [PORTED]

Pernet has NO --derivatives-root — its precomputed cut is contrast-level (--results-root).
The precomputed flow needs <results-root>/00_volumetric_GLM/ (Fig. 3b map + B3b) and
<results-root>/04_cross_validation/per_subject/ (Fig. 3b profile) present — ship via OSF.

--input-source raw regenerates that cut from the raw Edinburgh DataShare BIDS dataset
(--raw-root) via Stage-0 preprocessing (preprocessing/run_stage0.py: FSL + Nilearn
first-level GLM), writing it to --results-root, then runs the identical Stage-1 handlers.
Stage 0 is env-pinned (nilearn 0.10.4) + FSL-dependent and heavy (~30–60 min/subject);
it is a faithful port, not golden-mastered (docs/DESIGN.md §2.5/§6).

STATUS: all three Stage-1 figure lineages wired; Stage-0 raw path wired (faithful port).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import config

FIGURES = ("fig3b_map", "fig3b_profile", "figB3b_morans_i")
_HERE = Path(__file__).resolve().parent
_ANALYSIS = _HERE / "analysis"


def _load(module_filename: str):
    """Load a numbered analysis script (e.g. '01_group_analysis.py') as a module."""
    path = _ANALYSIS / module_filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fig3b_map(results_root: str, plots_root: str) -> None:
    """00 -> 01 group analysis -> 02 surface projection + Fig. 3b render."""
    root = str(results_root)
    _load("01_group_analysis.py").main(["--results-root", root])
    _load("02_surface_projection.py").main(["--results-root", root, "--plots-dir", str(plots_root)])


def _fig3b_profile(results_root: str, plots_root: str) -> None:
    """cv_04 group fROI definition -> cv_05 cross-validated responses + 2-bar plot."""
    root = str(results_root)
    _load("cv_04_group_froi_analysis.py").main(["--results-root", root])
    _load("cv_05_extract_responses_and_plot.py").main(["--results-root", root, "--plots-dir", str(plots_root)])


def _figB3b_morans_i(results_root: str, plots_root: str) -> None:
    """05 island Moran's I of the voice-selective map -> 06 3-bar comparison vs model (Fig. B3b)."""
    root = str(results_root)
    _load("05_island_morans_i.py").main(["--results-root", root])
    _load("06_island_morans_i_comparison.py").main(["--results-root", root, "--plots-dir", str(plots_root)])


DISPATCH = {
    "fig3b_map": _fig3b_map,
    "fig3b_profile": _fig3b_profile,
    "figB3b_morans_i": _figB3b_morans_i,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-source", choices=("precomputed", "raw"), default="precomputed")
    p.add_argument("--results-root", default=None, help="Contrast maps / CV GLMs (the precomputed cut) + intermediate outputs.")
    p.add_argument("--plots-root", default=str(_HERE / "plots"),
                   help="Directory for rendered figures (default: <dataset>/plots). Intermediate stat "
                        "maps/CSVs still go under --results-root.")
    p.add_argument("--raw-root", default=None, help="Edinburgh DataShare BIDS root (only for --input-source raw).")
    p.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # --results-root holds the precomputed cut (raw writes it, precomputed reads it) and
    # the figure outputs — required for both sources.
    if not args.results_root:
        raise SystemExit("--results-root is required (holds the precomputed cut + figure outputs).")

    if args.input_source == "raw":
        if not args.raw_root:
            raise SystemExit("--raw-root is required for --input-source raw (Edinburgh DataShare BIDS root).")
        raw_root = Path(args.raw_root)
        if not raw_root.exists():
            raise SystemExit(f"--raw-root not found: {raw_root}")
        print("=== Stage 0: raw -> precomputed cut (FSL + Nilearn; env-pinned, heavy) ===")
        # Import the Stage-0 package lazily (after validation) so the precomputed path and
        # the CLI arg surface stay import-light (no nibabel/nilearn/FSL). _HERE first on
        # sys.path disambiguates THIS dataset's `preprocessing` package.
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))
        from preprocessing.run_stage0 import build_precomputed_cut
        build_precomputed_cut(str(raw_root), args.results_root, figures=args.figures)

    for fig in args.figures:
        handler = DISPATCH.get(fig)
        if handler is None:
            raise NotImplementedError(
                f"{fig} is not yet ported (docs/DESIGN.md §9). Available: {sorted(DISPATCH)} "
                f"(dataset={config.DATASET})."
            )
        print(f"=== {fig} ===")
        handler(args.results_root, args.plots_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
