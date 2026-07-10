#!/usr/bin/env python3
"""fROI definition — top 10% t-voxels within each anatomical parcel (Branch A).

For each subject, parcel and run-split (even / odd), define a functional ROI by taking
the most-selective 10% of voxels of the split-averaged contrast z-map *within* the
anatomical parcel:

  parcel (template) --resample(nearest)--> functional grid
  mean z-map across the split's runs
  threshold = percentile(z within parcel, 100 - 10)
  fROI = { voxel in parcel : z >= threshold }

This is the "anatomical constraint + functional selection" method of Marvi et al.
(2025). Two masks are produced per ROI (split-even from runs 002/004, split-odd from
runs 001/003/005) so the cross-validation stage never double-dips.

Lineage (docs/DESIGN.md §2.4 / README §2):  06 first-level GLM -> batch_glm_splits ->
**define_frois** -> cross_validation -> extract -> Fig. A2.
  input : <derivatives-root>/<subj>/first_level_glm/effloc_{visual,auditory}_split-{even,odd}/
          run-*/..._space-MNI152NLin2009cAsym_res-2_zmap.nii.gz  (published GLM output)
          + anatomical parcels (<parcels-dir>/<cat>/[<hemi>.]<parcel>.nii.gz)
  output: <derivatives-root>/<subj>/frois/<cat>_<parcel>/
          <subj>_<parcel>[_<hemi>]_space-MNI152NLin2009cAsym_split-{even,odd}_froi.nii.gz

PORT NOTES vs dev-repo `src/batch_define_frois.py` + `src/emfl/roi/definition.py`
(@ ef1da34):
  * Faithful port. The fROI math is the vendored `emfl.roi.fROIDefiner.define_froi`
    (unchanged). This driver reproduces the *driver's* on-disk save layout
    (`frois/<cat>_<parcel>/...` — NOT the library `fROIDefiner.save_froi`, which is
    unused on disk).
  * Parameterized by `--derivatives-root` / `--parcels-dir` (were hard-coded dev paths).
  * `define_subject_frois()` is side-effect-free (returns the masks in memory) so the
    golden master can Dice-compare against the published masks without touching disk.
    Saving moves to `main()`.

DETERMINISM (docs/DESIGN.md §2.2/§6): the selection is pure numpy (`np.percentile` + boolean
mask); the one version-sensitive step is nilearn `resample_to_img(..., 'nearest')` for
the parcel. The golden master (tests/test_define_frois_golden.py) therefore asserts a
Dice overlap against the published masks with a calibrated-then-frozen tolerance rather
than bitwise equality.
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
)
from emfl.roi import fROIDefiner  # noqa: E402

SPLITS = ("even", "odd")

# Parcels whose masks are midline / single-hemisphere (mirror dev batch_define_frois.py).
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


def iter_froi_specs(parcel_cats=None, test_mode: bool = False):
    """Yield ``(category, parcel, hemisphere, split)`` for every fROI mask to define.

    Order matches the dev driver (category -> parcel -> hemisphere -> split). In
    ``test_mode`` only the first parcel of each category is used.
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
                for split in SPLITS:
                    yield cat, parcel_name, hemi, split


def froi_output_path(derivatives_root: Path, subject_id: str, category: str,
                     parcel_name: str, hemisphere, split: str,
                     space: str = DEFAULT_SPACE) -> Path:
    """On-disk path of a published fROI mask (the dev driver's save layout)."""
    hemi_str = f"_{hemisphere}" if hemisphere else ""
    return (Path(derivatives_root) / subject_id / "frois" / f"{category}_{parcel_name}"
            / f"{subject_id}_{parcel_name}{hemi_str}_space-{space}_split-{split}_froi.nii.gz")


def define_subject_frois(subject_id: str, parcels_dir, derivatives_root,
                         parcel_cats=None, space: str = DEFAULT_SPACE,
                         percentile: float = DEFAULT_FROI_PERCENTILE,
                         test_mode: bool = False, skip_missing: bool = True):
    """Define every fROI mask for one subject, in memory (no disk writes).

    Returns a list of records, each a dict with ``category``, ``parcel``, ``hemisphere``,
    ``split``, ``froi_img`` (nibabel image) and ``metadata``. Specs whose inputs are
    missing are skipped (``skip_missing=True``) or raised.
    """
    definer = fROIDefiner(
        parcels_dir=parcels_dir,
        derivatives_dir=derivatives_root,
        subject_id=subject_id,
        space=space,
    )
    records = []
    for cat, parcel_name, hemi, split in iter_froi_specs(parcel_cats, test_mode):
        try:
            froi_img, metadata = definer.define_froi(
                parcel_category=cat,
                parcel_name=parcel_name,
                hemisphere=hemi,
                percentile=percentile,
                run_split=split,
            )
        except FileNotFoundError:
            if skip_missing:
                continue
            raise
        records.append({
            "category": cat,
            "parcel": parcel_name,
            "hemisphere": hemi,
            "split": split,
            "froi_img": froi_img,
            "metadata": metadata,
        })
    return records


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derivatives-root", required=True,
                   help="fMRIPrep derivatives root (holds <subj>/first_level_glm/ and "
                        "receives <subj>/frois/).")
    p.add_argument("--parcels-dir", default=None,
                   help="Anatomical parcels dir. Default: emfl.config.get_parcels_dir() "
                        "(MARVI_PARCELS_DIR env / vendored data/PARCELS / dev fallback).")
    p.add_argument("--subjects", nargs="+", default=list(ALL_SUBJECTS),
                   help="Subject IDs (default: all 6 EMFL subjects).")
    p.add_argument("--parcels", nargs="+", default=None, dest="parcel_cats",
                   help="Parcel categories to process (default: all).")
    p.add_argument("--space", default=DEFAULT_SPACE, help="Analysis space (Branch A: MNI 2mm).")
    p.add_argument("--percentile", type=float, default=DEFAULT_FROI_PERCENTILE,
                   help="Top N%% of voxels to select (default: 10).")
    p.add_argument("--test", action="store_true",
                   help="Test mode: only the first parcel of each category.")
    return p


def main(argv=None) -> int:
    import nibabel as nib

    args = build_parser().parse_args(argv)
    parcels_dir = args.parcels_dir
    if parcels_dir is None:
        from emfl.config import get_parcels_dir
        parcels_dir = get_parcels_dir()

    derivatives_root = Path(args.derivatives_root)
    print("=" * 70)
    print("fROI DEFINITION (Branch A)")
    print(f"  subjects   : {', '.join(args.subjects)}")
    print(f"  parcels    : {args.parcel_cats if args.parcel_cats else 'all'}")
    print(f"  parcels-dir: {parcels_dir}")
    print(f"  derivatives: {derivatives_root}")
    print(f"  space      : {args.space}  | top {args.percentile}%  | splits {SPLITS}")
    print("=" * 70)

    total = 0
    for subject_id in args.subjects:
        records = define_subject_frois(
            subject_id=subject_id,
            parcels_dir=parcels_dir,
            derivatives_root=derivatives_root,
            parcel_cats=args.parcel_cats,
            space=args.space,
            percentile=args.percentile,
            test_mode=args.test,
        )
        for rec in records:
            out = froi_output_path(derivatives_root, subject_id, rec["category"],
                                   rec["parcel"], rec["hemisphere"], rec["split"], args.space)
            out.parent.mkdir(parents=True, exist_ok=True)
            nib.save(rec["froi_img"], out)
            total += 1
        print(f"  {subject_id}: saved {len(records)} fROI masks")
    print(f"Done. {total} masks written under {derivatives_root}/<subj>/frois/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
