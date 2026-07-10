#!/usr/bin/env python
"""Regenerate the model-side paper figures in one run (AlKhamissi & Mehrer et al., 2026).

Thin orchestrator (mirrors fMRI/make_all_figures.py): dispatches each per-figure script
under scripts/figures/ in the chosen mode.

    # default: plot from the hosted precomputed cut (no GPU / no stimuli)
    python download_precomputed.py --dest _precomputed_cut
    python make_all_figures.py --input-source precomputed --derivatives-root _precomputed_cut

    # recompute from the HuggingFace model + stimuli (needs a GPU; some panels need private stimuli)
    python make_all_figures.py --input-source raw --derivatives-root results --model epfl-neuroai/topo-omni

    python make_all_figures.py --figures 3,4 --dry-run     # subset / preview commands

Panels land in <out>/figure_<id>/. This covers the *model* side; the brain-side figures live
under fMRI/ (see fMRI/make_all_figures.py).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FIG_DIR = _HERE / "scripts" / "figures"
# Figures with a reproduction script. 2c/7 are the SpaceTop-discovery panels.
FIGURES = ["2", "3", "4", "5", "6", "7"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-source", choices=("precomputed", "raw"), default="precomputed",
                   help="Plot from the hosted precomputed cut (default) or recompute from the HF model.")
    p.add_argument("--derivatives-root", default=os.getenv("SAVE_DIR"),
                   help="Cut root (== SAVE_DIR): where the precomputed intermediates live "
                        "(precomputed) or where recomputed outputs are written (raw).")
    p.add_argument("--out", default=str(_HERE / "figures_out"),
                   help="Base output dir; each figure writes to <out>/figure_<id>/.")
    p.add_argument("--model", default=None, help="Raw mode: HF id or local checkpoint dir.")
    p.add_argument("--figures", default=None, help="Comma-separated subset (e.g. 3,4). Default: all.")
    p.add_argument("--python", default=sys.executable, help="Interpreter for the per-figure scripts.")
    p.add_argument("--dry-run", action="store_true", help="Print the per-figure commands and exit.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    figs = [f.strip() for f in args.figures.split(",")] if args.figures else list(FIGURES)

    if args.input_source == "precomputed" and not args.derivatives_root:
        raise SystemExit("--input-source precomputed needs --derivatives-root <cut> (or set $SAVE_DIR). "
                         "Run download_precomputed.py first.")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(_HERE), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

    failures = []
    for fig in figs:
        script = _FIG_DIR / f"figure_{fig}.py"
        if not script.exists():
            print(f"  skip figure {fig}: no scripts/figures/figure_{fig}.py")
            continue
        cmd = [args.python, str(script), "--input-source", args.input_source, "--out", args.out]
        if args.derivatives_root:
            cmd += ["--derivatives-root", args.derivatives_root]
        if args.model:
            cmd += ["--model", args.model]
        print(f"\n=== figure {fig} ===\n    {' '.join(cmd)}")
        if args.dry_run:
            continue
        if subprocess.run(cmd, cwd=str(_HERE), env=env).returncode != 0:
            failures.append(fig)
            print(f"    ✗ figure {fig} failed")

    if args.dry_run:
        print("\n(dry run — nothing executed)")
        return 0
    if failures:
        print("\nFAILED figures: " + ", ".join(failures))
        return 1
    print(f"\nAll requested figures reproduced -> {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
