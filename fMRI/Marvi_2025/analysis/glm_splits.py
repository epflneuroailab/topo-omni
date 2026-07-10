#!/usr/bin/env python3
"""Split-runs first-level EMFL GLM (all / even / odd) — the raw-path cut for Branch A.

Re-runs the first-level GLM on independent run subsets so fROIs are defined and validated
without double-dipping (Marvi et al. 2025 methodology):
  all  = runs 001,002,003,004,005   ->  first_level_glm/effloc_{modality}/
  even = runs 002,004               ->  first_level_glm/effloc_{modality}_split-even/
  odd  = runs 001,003,005           ->  first_level_glm/effloc_{modality}_split-odd/

The **even/odd split cut is what the Branch-A Tier-1 goldens consume**: define_frois reads
the split zmaps, cross_validation compares even-vs-odd fROIs, and extract_condition_responses
reads the split per-condition effect maps. This module is the `--input-source raw` producer
for that cut.

Lineage (docs/DESIGN.md §2.4 / README §6b):  06 first-level GLM -> **batch_glm_splits** ->
define_frois -> cross_validation -> extract_condition_responses -> Fig. A2.

PORT NOTES vs dev `src/batch_glm_splits.py` + `src/emfl/glm/first_level.py` (@ ef1da34):
  * Thin faithful port. Reuses the shared engine wrappers from `first_level_glm`
    (`build_analyzer` threads `--raw-root` to the engine's events root; `run_subject_run`
    runs visual+auditory for one run) — one analyzer per (subject, split) with the engine's
    ``run_split`` set, exactly as dev. The zmap-restore + events-root parameterization live
    in the vendored engine / `first_level_glm` (see their PORT NOTES).
  * The engine's ``run_split`` controls both the selected runs and the output-dir suffix
    (``effloc_{modality}_split-{even,odd}``; no suffix for 'all'), so no path logic is
    duplicated here.

DETERMINISM / TESTING (docs/DESIGN.md §6): heavy nilearn GLM — **NOT golden-mastered** (see
`first_level_glm` header). Deliverable = faithful port + raw-dispatch smoke test. Run heavy
fits via SLURM bigmem (README §1), not the login node.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `Marvi_2025/` importable so `emfl` + `config` (+ sibling analysis modules) resolve.
_DATASET_DIR = Path(__file__).resolve().parent.parent
if str(_DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASET_DIR))

from emfl.config import ALL_SUBJECTS, DEFAULT_SPACE, RUN_SPLITS  # noqa: E402

from analysis.first_level_glm import build_analyzer, run_subject_run  # noqa: E402

DEFAULT_SPLITS = ("all", "even", "odd")


def run_glm_splits(subjects, derivatives_root, raw_root=None, splits=DEFAULT_SPLITS,
                   space: str = DEFAULT_SPACE, smoothing_fwhm: float = 3.0,
                   save_outputs: bool = True) -> list:
    """Run the first-level GLM for each subject x split. Returns per-(subject,run,split) records.

    For each split, one analyzer is built with the engine's ``run_split`` set (which selects
    the runs and the output-dir suffix), then every run in that split is fit for both
    modalities.
    """
    results = []
    for subject_id in subjects:
        for split in splits:
            if split not in RUN_SPLITS:
                raise ValueError(f"unknown split {split!r}; must be one of {list(RUN_SPLITS)}")
            selected_runs = RUN_SPLITS[split]
            analyzer = build_analyzer(
                derivatives_root, subject_id, space=space, smoothing_fwhm=smoothing_fwhm,
                run_split=split, raw_root=raw_root)
            print(f"\n{'#'*70}\n# {subject_id} | split {split} | runs {selected_runs}\n{'#'*70}")
            for run in selected_runs:
                results.append(
                    run_subject_run(analyzer, subject_id, run, save_outputs=save_outputs))
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derivatives-root", required=True,
                   help="fMRIPrep derivatives root (holds <subj>/func/; split GLM maps "
                        "written under <subj>/first_level_glm/effloc_*_split-*/).")
    p.add_argument("--raw-root", default=None,
                   help="Raw BIDS root for event TSVs. If omitted, derived from "
                        "--derivatives-root (dev hack).")
    p.add_argument("--subjects", nargs="+", default=list(ALL_SUBJECTS),
                   help="Subject IDs (default: all 6 EMFL subjects).")
    p.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS),
                   choices=list(DEFAULT_SPLITS),
                   help="Run splits to process (default: all even odd). Branch A fROI "
                        "cross-validation needs even + odd.")
    p.add_argument("--space", default=DEFAULT_SPACE,
                   choices=["MNI152NLin2009cAsym", "T1w", "fsaverage5", "fsnative"],
                   help="Analysis space (Branch A: MNI 2mm).")
    p.add_argument("--smoothing", type=float, default=3.0,
                   help="Smoothing FWHM in mm (default 3.0).")
    p.add_argument("--no-save", action="store_true",
                   help="Do not write maps to disk (dry validation).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    print("=" * 70)
    print("SPLIT-RUNS EMFL GLM (all/even/odd) — Branch A raw-path cut")
    print(f"  subjects   : {', '.join(args.subjects)}")
    print(f"  splits     : {', '.join(args.splits)}")
    print(f"  derivatives: {args.derivatives_root}")
    print(f"  raw (events): {args.raw_root or '(derived from derivatives)'}")
    print(f"  space      : {args.space}   smoothing: {args.smoothing}mm")
    print("=" * 70)

    results = run_glm_splits(
        subjects=args.subjects, derivatives_root=args.derivatives_root,
        raw_root=args.raw_root, splits=args.splits, space=args.space,
        smoothing_fwhm=args.smoothing, save_outputs=not args.no_save)

    n_ok = sum(1 for r in results
               if r.get("visual_status") == "success" and r.get("auditory_status") == "success")
    print(f"\n{n_ok}/{len(results)} subject-run(-split) fits fully succeeded.")
    failed = [r for r in results if r["errors"]]
    for r in failed:
        print(f"  ! {r['subject']} split {r['run_split']} run {r['run']}: {'; '.join(r['errors'])}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
