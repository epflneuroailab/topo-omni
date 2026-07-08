#!/usr/bin/env python3
"""fROI cross-validation — split-half reliability of the functional ROIs (Branch A).

For each subject / parcel / hemisphere, quantify how well the fROIs generalize across
independent run splits (never double-dipping — the mask and the responses come from
disjoint runs):

  1. odd-split fROI  --extract-->  responses from EVEN runs (002, 004)
  2. even-split fROI --extract-->  responses from ODD runs (001, 003, 005)
  3. Dice(odd-mask, even-mask)                    -> spatial overlap of the two masks
  4. pearsonr(z_even, z_odd) over anatomical parcel -> voxel-wise pattern reliability
     (paper Supp. Fig. S8)
  5. mean preferred-contrast response per split    -> response reliability

This is the reliability analysis of Marvi et al. (2025) (Left FFA Dice ≈ 0.735 in the
paper; high spatial-pattern correlation r > 0.7).

Lineage (docs/DESIGN.md §2.4 / README §2):  06 first-level GLM -> batch_glm_splits ->
define_frois -> **cross_validation** -> extract -> Fig. A2.
  input : <derivatives-root>/<subj>/frois/<cat>_<parcel>/..._split-{even,odd}_froi.nii.gz
          (published fROI masks — the define_frois output)
          + <subj>/first_level_glm/effloc_{visual,auditory}_split-{even,odd}/run-*/
            ..._space-MNI152NLin2009cAsym_res-2_zmap.nii.gz  (published GLM output)
          + anatomical parcels (for the voxel-wise pattern correlation)
  output: <derivatives-root>/<subj>/roi_cross_validation/
          <subj>_<roi_label>_{even_from_odd,odd_from_even}.csv   (per-ROI responses)
          + a details CSV of the scalar reliability metrics (Dice, spatial-corr, means).

PORT NOTES vs dev-repo `src/batch_cross_validation.py` + `src/emfl/roi/validation.py`
(@ ef1da34):
  * Faithful port. The reliability math is the vendored `emfl.roi.CrossValidationAnalyzer`
    primitives (`load_existing_froi`, `extract_roi_responses`, `compute_dice_coefficient`,
    `compute_spatial_pattern_correlation`) — all unchanged and side-effect-free. This
    driver re-orchestrates them so the whole compute is side-effect-free (returns the
    records + response frames in memory); the CSV writes move to `main()`. Only the
    library's `cross_validate_roi` tail wrote to disk, and it is *not* called here.
  * Parameterized by `--derivatives-root` / `--parcels-dir` (were hard-coded dev paths).
  * The details-CSV schema and the per-ROI response-CSV schema reproduce the dev driver's
    on-disk layout exactly, so the golden master can compare column-for-column.

DETERMINISM (docs/DESIGN.md §2.2/§6): Dice is pure numpy over the two published masks; the mean
responses are mean-in-mask of the published z-maps (pure numpy). The one version-sensitive
step is nilearn `resample_to_img(parcel, ..., 'nearest')` inside the spatial-pattern
correlation (only when the parcel grid differs from the functional grid). The golden master
(tests/test_cross_validation_golden.py) therefore pins the dev-computed scalars with a
calibrated-then-frozen tolerance rather than asserting bitwise equality.
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
    DEFAULT_FROI_PERCENTILE,
    PARCEL_CATEGORIES,
    PARCEL_CONTRAST_MAP,
)
from emfl.roi import CrossValidationAnalyzer  # noqa: E402

# Parcels whose masks are midline / single-hemisphere (mirror dev batch_cross_validation.py,
# identical to define_frois).
_BILATERAL_TOM = {"mmpfc", "vmpfc", "dmpfc", "pc"}


def hemispheres_for(parcel_category: str, parcel_name: str) -> list:
    """Hemisphere list for a parcel (``[None]`` means one bilateral/midline mask).

    Matches the dev driver exactly: default L+R; ToM midline parcels + speech are a
    single bilateral mask; VWFA is left-only.
    """
    if parcel_category == "tom" and parcel_name in _BILATERAL_TOM:
        return [None]
    if parcel_category == "speech":
        return [None]
    if parcel_category == "vwfa":
        return ["lh"]
    return ["lh", "rh"]


def iter_cv_specs(parcel_cats=None, test_mode: bool = False):
    """Yield ``(category, parcel, hemisphere)`` for every ROI to cross-validate.

    Order matches the dev driver (category -> parcel -> hemisphere). In ``test_mode``
    only the first parcel of each category is used.
    """
    if parcel_cats is None:
        parcel_cats = list(PARCEL_CATEGORIES.keys())
    for cat in parcel_cats:
        if cat not in PARCEL_CATEGORIES:
            continue
        parcels = PARCEL_CATEGORIES[cat]
        if test_mode:
            parcels = parcels[:1]
        for parcel_name in parcels:
            for hemi in hemispheres_for(cat, parcel_name):
                yield cat, parcel_name, hemi


def roi_label_for(parcel_name: str, hemisphere) -> str:
    """ROI label used in the response-CSV filenames (dev driver convention)."""
    return f"{hemisphere}_{parcel_name}" if hemisphere else parcel_name


def response_csv_paths(derivatives_root: Path, subject_id: str, roi_label: str):
    """(even_from_odd, odd_from_even) response-CSV paths (the dev driver's save layout)."""
    out_dir = Path(derivatives_root) / subject_id / "roi_cross_validation"
    return (out_dir / f"{subject_id}_{roi_label}_even_from_odd.csv",
            out_dir / f"{subject_id}_{roi_label}_odd_from_even.csv")


def cross_validate_roi(analyzer: CrossValidationAnalyzer, parcel_category: str,
                       parcel_name: str, hemisphere):
    """Cross-validate one ROI, in memory (no disk writes).

    Re-orchestrates the vendored analyzer's side-effect-free primitives. Returns
    ``(record, responses_even, responses_odd)`` where ``record`` is the scalar-metrics
    dict (details-CSV row, minus the save-time response-CSV paths) and the two frames are
    the per-ROI response extractions. Raises ``FileNotFoundError`` if a split mask is
    missing (caller decides whether to skip).
    """
    import nibabel as nib
    import numpy as np

    roi_label = roi_label_for(parcel_name, hemisphere)

    # STEP 1-2: odd-split mask -> responses from EVEN runs (mask & data disjoint).
    froi_odd_path, froi_odd_voxels = analyzer.load_existing_froi(
        parcel_category, parcel_name, hemisphere, run_split="odd")
    responses_even = analyzer.extractor.extract_roi_responses(
        roi_mask_path=froi_odd_path, run_split="even")

    # STEP 3-4: even-split mask -> responses from ODD runs.
    froi_even_path, froi_even_voxels = analyzer.load_existing_froi(
        parcel_category, parcel_name, hemisphere, run_split="even")
    responses_odd = analyzer.extractor.extract_roi_responses(
        roi_mask_path=froi_even_path, run_split="odd")

    record = {
        "subject": analyzer.subject_id,
        "parcel_category": parcel_category,
        "parcel_name": parcel_name,
        "hemisphere": hemisphere,
        "roi_label": roi_label,
        "percentile": analyzer.percentile,
        "space": analyzer.space,
        "froi_odd_voxels": froi_odd_voxels,
        "froi_odd_path": froi_odd_path,
        "responses_even_from_odd": len(responses_even),
        "froi_even_voxels": froi_even_voxels,
        "froi_even_path": str(froi_even_path),
        "responses_odd_from_even": len(responses_odd),
    }

    # STEP 5a: Dice overlap of the two masks (pure numpy).
    froi_odd_data = nib.load(froi_odd_path).get_fdata()
    froi_even_data = nib.load(froi_even_path).get_fdata()
    record["dice_coefficient"] = float(
        analyzer.compute_dice_coefficient(froi_odd_data, froi_even_data))

    # STEP 5b: mean preferred-contrast response per split.
    contrast_name = PARCEL_CONTRAST_MAP.get(parcel_name)
    if contrast_name:
        even_pref = responses_even[responses_even["contrast"] == contrast_name]["response"].values
        odd_pref = responses_odd[responses_odd["contrast"] == contrast_name]["response"].values
        if len(even_pref) > 0 and len(odd_pref) > 0:
            even_mean = np.mean(even_pref)
            odd_mean = np.mean(odd_pref)
            record["even_mean_response"] = float(even_mean)
            record["odd_mean_response"] = float(odd_mean)
            record["mean_difference"] = float(abs(even_mean - odd_mean))

    # STEP 5c: voxel-wise spatial pattern correlation over the anatomical parcel.
    try:
        spatial_corr, spatial_pval = analyzer.compute_spatial_pattern_correlation(
            parcel_category=parcel_category, parcel_name=parcel_name,
            hemisphere=hemisphere, contrast_name=contrast_name)
        record["spatial_correlation"] = float(spatial_corr)
        record["spatial_pval"] = float(spatial_pval)
    except Exception as e:  # noqa: BLE001 - match dev: soft-fail, record None
        print(f"  x Could not compute spatial correlation for {roi_label}: {e}")
        record["spatial_correlation"] = None
        record["spatial_pval"] = None

    return record, responses_even, responses_odd


def cross_validate_subject(subject_id: str, parcels_dir, derivatives_root,
                           parcel_cats=None, space: str = DEFAULT_SPACE,
                           percentile: float = DEFAULT_FROI_PERCENTILE,
                           test_mode: bool = False, skip_missing: bool = True):
    """Cross-validate every ROI for one subject, in memory (no disk writes).

    Returns a list of records, each a dict with the scalar metrics plus ``responses_even``
    and ``responses_odd`` frames. ROIs whose split masks are missing are skipped
    (``skip_missing=True``) or raised.
    """
    analyzer = CrossValidationAnalyzer(
        parcels_dir=parcels_dir,
        derivatives_dir=derivatives_root,
        subject_id=subject_id,
        space=space,
        percentile=percentile,
    )
    records = []
    for cat, parcel_name, hemi in iter_cv_specs(parcel_cats, test_mode):
        try:
            record, resp_even, resp_odd = cross_validate_roi(
                analyzer, cat, parcel_name, hemi)
        except FileNotFoundError:
            if skip_missing:
                continue
            raise
        record["responses_even"] = resp_even
        record["responses_odd"] = resp_odd
        records.append(record)
    return records


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derivatives-root", required=True,
                   help="fMRIPrep derivatives root (holds <subj>/frois/, "
                        "<subj>/first_level_glm/; receives <subj>/roi_cross_validation/).")
    p.add_argument("--parcels-dir", default=None,
                   help="Anatomical parcels dir (for the spatial-pattern correlation). "
                        "Default: emfl.config.get_parcels_dir().")
    p.add_argument("--subjects", nargs="+", default=list(ALL_SUBJECTS),
                   help="Subject IDs (default: all 6 EMFL subjects).")
    p.add_argument("--parcels", nargs="+", default=None, dest="parcel_cats",
                   help="Parcel categories to process (default: all).")
    p.add_argument("--space", default=DEFAULT_SPACE, help="Analysis space (Branch A: MNI 2mm).")
    p.add_argument("--percentile", type=float, default=DEFAULT_FROI_PERCENTILE,
                   help="fROI selection percentile the masks were defined at (default: 10).")
    p.add_argument("--details-csv", default=None,
                   help="Where to write the scalar-metrics details CSV. Default: "
                        "<derivatives-root>/cross_validation_details.csv.")
    p.add_argument("--test", action="store_true",
                   help="Test mode: only the first parcel of each category.")
    return p


def main(argv=None) -> int:
    import pandas as pd

    args = build_parser().parse_args(argv)
    parcels_dir = args.parcels_dir
    if parcels_dir is None:
        from emfl.config import get_parcels_dir
        parcels_dir = get_parcels_dir()

    derivatives_root = Path(args.derivatives_root)
    print("=" * 70)
    print("fROI CROSS-VALIDATION (Branch A)")
    print(f"  subjects   : {', '.join(args.subjects)}")
    print(f"  parcels    : {args.parcel_cats if args.parcel_cats else 'all'}")
    print(f"  parcels-dir: {parcels_dir}")
    print(f"  derivatives: {derivatives_root}")
    print(f"  space      : {args.space}  | top {args.percentile}%")
    print("=" * 70)

    detail_rows = []
    for subject_id in args.subjects:
        records = cross_validate_subject(
            subject_id=subject_id,
            parcels_dir=parcels_dir,
            derivatives_root=derivatives_root,
            parcel_cats=args.parcel_cats,
            space=args.space,
            percentile=args.percentile,
            test_mode=args.test,
        )
        out_dir = derivatives_root / subject_id / "roi_cross_validation"
        out_dir.mkdir(parents=True, exist_ok=True)
        for rec in records:
            even_path, odd_path = response_csv_paths(
                derivatives_root, subject_id, rec["roi_label"])
            rec["responses_even"].to_csv(even_path, index=False)
            rec["responses_odd"].to_csv(odd_path, index=False)
            row = {k: v for k, v in rec.items()
                   if k not in ("responses_even", "responses_odd")}
            row["responses_even_path"] = str(even_path)
            row["responses_odd_path"] = str(odd_path)
            detail_rows.append(row)
        print(f"  {subject_id}: cross-validated {len(records)} ROIs")

    details_csv = (Path(args.details_csv) if args.details_csv
                   else derivatives_root / "cross_validation_details.csv")
    pd.DataFrame(detail_rows).to_csv(details_csv, index=False)

    if detail_rows:
        dice = [r["dice_coefficient"] for r in detail_rows if r.get("dice_coefficient") is not None]
        corr = [r["spatial_correlation"] for r in detail_rows
                if r.get("spatial_correlation") is not None]
        import numpy as np
        print(f"Done. {len(detail_rows)} ROIs -> {details_csv}")
        if dice:
            print(f"  mean Dice           : {np.mean(dice):.3f} +/- {np.std(dice):.3f}")
        if corr:
            print(f"  mean spatial pattern r: {np.mean(corr):.3f} +/- {np.std(corr):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
