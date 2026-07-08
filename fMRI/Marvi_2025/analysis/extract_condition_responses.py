#!/usr/bin/env python3
"""Cross-validated per-condition fROI responses — THE Fig-A2 data (Branch A).

For every subject / parcel / hemisphere, extract the mean per-condition effect (beta) from
the functional ROIs, cross-validated so the mask and the data never come from the same runs:

  even-split fROI  --extract-->  betas from ODD runs (001, 003, 005)   -> even_beta
  odd-split  fROI  --extract-->  betas from EVEN runs (002, 004)        -> odd_beta
  mean_beta = (even_beta + odd_beta) / 2                                per condition

This reproduces the per-condition response profiles behind Fig. A2 of Marvi et al. (2025)
(the 3x5 ROI-category x condition grid, `replication_of_fig_4_from_Marvi_2025_with_indiv.svg`). The 10
conditions are the 5 visual (faces, bodies, scenes, objects, words_scr_objects) and 5
auditory (false_belief, false_photo, nonwords, quilted_speech, math) localizer conditions.

Lineage (docs/DESIGN.md §2.4 / README §6b):  06 first-level GLM -> batch_glm_splits ->
define_frois -> **extract_condition_responses** -> Fig. A2.
  input : <derivatives-root>/<subj>/frois/<cat>_<parcel>/..._split-{even,odd}_froi.nii.gz
          (published fROI masks — the define_frois output)
          + <subj>/first_level_glm/effloc_{visual,auditory}_split-{even,odd}/run-*/
            ..._{condition}_space-MNI152NLin2009cAsym_res-2_effect.nii.gz  (published GLM)
  output: <results-root>/condition_responses_details.csv   (THE Fig-A2 details table)
          + condition_responses_summary.csv                (per-ROI/condition mean/std/sem/n)

PORT NOTES vs dev-repo `src/batch_extract_condition_responses.py` +
`src/emfl/roi/extraction.py` (@ ef1da34):
  * Faithful port. The per-condition beta math is the vendored
    `emfl.roi.ROIResponseExtractor.extract_condition_responses` (unchanged, side-effect-free:
    globs the per-condition effect maps and takes `mean(effect_data[mask>0])` per run, then
    the driver averages runs -> even_beta/odd_beta and (even+odd)/2 -> mean_beta).
  * `extract_subject_responses()` is already side-effect-free (returns a list of record
    dicts, only prints). Kept it that way, parameterized `--derivatives-root`; the CSV
    writes move to `main()`.
  * Dev's `main()` also renders the Fig-A2 SVG/PNG (seaborn/matplotlib) and a group summary.
    The plotting is render-dependent and NOT golden-mastered (README §6b), so it is NOT
    ported here — this module produces the numeric details+summary tables only. A separate
    plotting entry point can consume the details CSV later if needed.
  * Hemisphere rule mirrors dev EXACTLY (category-only): vwfa -> lh, speech -> bilateral
    (one mask), else lh+rh. This intentionally differs from define_frois/cross_validation
    (which treat the ToM midline parcels pc/dmpfc/mmpfc/vmpfc as one bilateral mask): here
    those parcels are looked up as lh/rh, miss on disk, and are silently skipped — so only
    tpj survives in the ToM category, reproducing the published 50-ROI / 500-row layout.

DETERMINISM (docs/DESIGN.md §2.2/§6): every beta is a pure nibabel-load + numpy mean-in-mask of a
published effect map on the fROI's own grid — no resampling, no nilearn. So the Fig-A2 table
reproduces bitwise (up to float summation order); the golden master
(tests/test_extract_condition_responses_golden.py) pins it with a tight, frozen tolerance
(the Pernet cv_05 regime).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `Marvi_2025/` importable so `emfl` + `config` resolve when run as a script.
_DATASET_DIR = Path(__file__).resolve().parent.parent
if str(_DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASET_DIR))

from emfl.config import (  # noqa: E402
    ALL_SUBJECTS,
    DEFAULT_SPACE,
    PARCEL_CATEGORIES,
)
from emfl.roi import ROIResponseExtractor  # noqa: E402

# The 5 visual localizer conditions; everything else is auditory (dev inline convention).
_VISUAL_CONDITIONS = {"faces", "objects", "scenes", "bodies", "words_scr_objects"}


def hemispheres_for(parcel_category: str) -> list:
    """Hemisphere list for a parcel category (dev extraction rule — category-only).

    vwfa -> left only; speech -> one bilateral mask (``None``); everything else -> L+R.
    NOTE: unlike define_frois/cross_validation this does *not* special-case the ToM midline
    parcels — they are looked up as lh/rh and skipped when the mask is absent on disk.
    """
    if parcel_category == "vwfa":
        return ["lh"]
    if parcel_category == "speech":
        return [None]
    return ["lh", "rh"]


def froi_mask_paths(derivatives_root: Path, subject_id: str, parcel_category: str,
                    parcel_name: str, hemisphere, space: str = DEFAULT_SPACE):
    """(even_mask, odd_mask) paths for one ROI (dev driver's filename convention)."""
    hemi_str = f"_{hemisphere}" if hemisphere else ""
    mask_dir = Path(derivatives_root) / subject_id / "frois" / f"{parcel_category}_{parcel_name}"
    even = mask_dir / f"{subject_id}_{parcel_name}{hemi_str}_space-{space}_split-even_froi.nii.gz"
    odd = mask_dir / f"{subject_id}_{parcel_name}{hemi_str}_space-{space}_split-odd_froi.nii.gz"
    return even, odd


def _modality_of(condition: str) -> str:
    return "visual" if condition in _VISUAL_CONDITIONS else "auditory"


def extract_subject_responses(subject_id: str, derivatives_root, parcel_cats=None,
                              space: str = DEFAULT_SPACE, test_mode: bool = False):
    """Cross-validated per-condition responses for one subject, in memory (no disk writes).

    Faithful port of dev ``extract_subject_responses``. Returns a list of record dicts with
    columns: subject, parcel_category, parcel_name, hemisphere, roi_label, condition,
    modality, even_beta, odd_beta, mean_beta. ROIs whose split masks are missing on disk are
    skipped (matches dev).
    """
    derivatives_root = Path(derivatives_root)
    if parcel_cats is None:
        parcel_cats = list(PARCEL_CATEGORIES.keys())

    extractor = ROIResponseExtractor(
        derivatives_dir=derivatives_root, subject_id=subject_id, space=space)

    results = []
    for cat in parcel_cats:
        if cat not in PARCEL_CATEGORIES:
            continue
        parcel_names = PARCEL_CATEGORIES[cat]
        if test_mode:
            parcel_names = parcel_names[:1]

        for parcel_name in parcel_names:
            for hemi in hemispheres_for(cat):
                roi_label = f"{hemi}_{parcel_name}" if hemi else parcel_name
                even_mask, odd_mask = froi_mask_paths(
                    derivatives_root, subject_id, cat, parcel_name, hemi, space)

                if not even_mask.exists() or not odd_mask.exists():
                    print(f"  - {roi_label}: fROI masks not found")
                    continue

                print(f"\n  Extracting: {roi_label}")
                # Cross-validated: even mask -> odd data; odd mask -> even data.
                try:
                    even_df = extractor.extract_condition_responses(
                        roi_mask_path=even_mask, run_split="odd")
                except Exception as e:  # noqa: BLE001 - match dev: skip ROI on error
                    print(f"    x Error extracting from even mask: {e}")
                    continue
                try:
                    odd_df = extractor.extract_condition_responses(
                        roi_mask_path=odd_mask, run_split="even")
                except Exception as e:  # noqa: BLE001
                    print(f"    x Error extracting from odd mask: {e}")
                    continue

                for condition in even_df["condition"].unique():
                    even_beta = even_df[even_df["condition"] == condition]["beta"].mean()
                    odd_beta = odd_df[odd_df["condition"] == condition]["beta"].mean()
                    mean_beta = (even_beta + odd_beta) / 2.0
                    results.append({
                        "subject": subject_id,
                        "parcel_category": cat,
                        "parcel_name": parcel_name,
                        "hemisphere": hemi if hemi else "bilateral",
                        "roi_label": roi_label,
                        "condition": condition,
                        "modality": _modality_of(condition),
                        "even_beta": even_beta,
                        "odd_beta": odd_beta,
                        "mean_beta": mean_beta,
                    })
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derivatives-root", required=True,
                   help="fMRIPrep derivatives root (holds <subj>/frois/ and "
                        "<subj>/first_level_glm/).")
    p.add_argument("--results-root", default=None,
                   help="Where to write the details + summary CSVs. "
                        "Default: <derivatives-root>.")
    p.add_argument("--subjects", nargs="+", default=list(ALL_SUBJECTS),
                   help="Subject IDs (default: all 6 EMFL subjects).")
    p.add_argument("--parcels", nargs="+", default=None, dest="parcel_cats",
                   choices=list(PARCEL_CATEGORIES.keys()),
                   help="Parcel categories to process (default: all).")
    p.add_argument("--space", default=DEFAULT_SPACE, help="Analysis space (Branch A: MNI 2mm).")
    p.add_argument("--details-csv", default=None,
                   help="Explicit path for the details CSV. Default: "
                        "<results-root>/condition_responses_details.csv.")
    p.add_argument("--test", action="store_true",
                   help="Test mode: only the first parcel of each category.")
    return p


def main(argv=None) -> int:
    import pandas as pd

    args = build_parser().parse_args(argv)
    derivatives_root = Path(args.derivatives_root)
    results_root = Path(args.results_root) if args.results_root else derivatives_root
    results_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PER-CONDITION fROI RESPONSES — Fig. A2 (Branch A)")
    print(f"  subjects   : {', '.join(args.subjects)}")
    print(f"  parcels    : {args.parcel_cats if args.parcel_cats else 'all'}")
    print(f"  derivatives: {derivatives_root}")
    print(f"  results    : {results_root}")
    print(f"  space      : {args.space}")
    print("=" * 70)

    all_results = []
    for subject_id in args.subjects:
        recs = extract_subject_responses(
            subject_id=subject_id, derivatives_root=derivatives_root,
            parcel_cats=args.parcel_cats, space=args.space, test_mode=args.test)
        all_results.extend(recs)
        print(f"  {subject_id}: {len(recs)} condition responses")

    if not all_results:
        print("No responses extracted. Exiting.")
        return 1

    df = pd.DataFrame(all_results)
    details_csv = (Path(args.details_csv) if args.details_csv
                   else results_root / "condition_responses_details.csv")
    df.to_csv(details_csv, index=False)
    print(f"\nDetails -> {details_csv}  ({len(df)} rows, {df['roi_label'].nunique()} ROIs)")

    # Per-ROI/condition summary (dev groupby: mean/std/sem/count of mean_beta).
    summary = df.groupby(
        ["roi_label", "condition", "modality", "parcel_category", "parcel_name", "hemisphere"]
    ).agg({"mean_beta": ["mean", "std", "sem", "count"]}).reset_index()
    summary.columns = ["_".join(c).strip("_") if c[1] else c[0]
                       for c in summary.columns.values]
    summary_csv = results_root / "condition_responses_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Summary -> {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
