#!/usr/bin/env python3
"""First-level EMFL GLM, per run — the raw-path GLM step (Branch A, docs/DESIGN.md §2.4).

Runs the nilearn first-level GLM for each subject x run x modality (visual/auditory) and
writes the per-run contrast + per-condition effect maps under
``<derivatives-root>/<subj>/first_level_glm/effloc_{visual,auditory}/run-<run>/``. This is
the `--input-source raw` producer for the maps that the precomputed Branch-A tier ships:
fROI definition, cross-validation, and per-condition response extraction all read these.

Lineage (docs/DESIGN.md §2.4 / README §6b):  **06 first-level GLM** -> batch_glm_splits ->
define_frois -> cross_validation -> extract_condition_responses -> Fig. A2.
  input : <derivatives-root>/<subj>/func/  (fMRIPrep preproc BOLD + confounds, MNI 2mm)
          + events from <raw-root>/<subj>/func/*_task-effloc{Visual,Auditory}Conditions_*
  output: <derivatives-root>/<subj>/first_level_glm/effloc_{visual,auditory}/run-<run>/
          ..._res-2_{zmap,beta,tmap,pval}.nii.gz  +  ..._res-2_effect.nii.gz (per condition)

PORT NOTES vs dev `src/06_batch_first_level_glm.py` + `src/emfl/glm/first_level.py` (@ ef1da34):
  * Thin faithful port of the dev batch driver. The GLM math is the vendored
    `emfl.glm.EFMLOCFirstLevelGLM` (nilearn FirstLevelModel: t_r from BOLD header,
    noise='ar1', hrf='spm', drift='polynomial' order 1, high_pass=0.01, smoothing 3mm).
  * PATHS PARAMETERIZED (docs/DESIGN.md §7): dev hardcoded the derivatives dir and derived the
    events tree by `str(derivatives).replace('derivatives','orig_data')`. Here
    ``--derivatives-root`` and ``--raw-root`` (events) are explicit; `--raw-root` is
    threaded to the engine's new ``orig_data_dir`` param (see its RELEASE PORT NOTE).
  * ZMAP RESTORED (docs/DESIGN.md §7 / README §6b): the dev volumetric engine dropped the
    z-score contrast map, but every Branch-A reader consumes `..._res-2_zmap.nii.gz`. The
    vendored engine now also saves the zmap (see `_run_effloc_glm_volumetric`), so this
    raw path is functional end-to-end.

DETERMINISM / TESTING (docs/DESIGN.md §6): this is the heavy nilearn GLM fit — it is **NOT**
golden-mastered (not bitwise reproducible across nilearn/machines, and the published cut
is a historical accretion across engine versions). Stage-0-style deliverable: faithful
port + parameterized paths + a raw-dispatch smoke test (`tests/test_raw_glm_smoke.py`).
Run the heavy fits via SLURM bigmem (README §1 compute caution), not the login node.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `Marvi_2025/` importable so `emfl` + `config` resolve when run as a script.
_DATASET_DIR = Path(__file__).resolve().parent.parent
if str(_DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASET_DIR))

from emfl.config import ALL_RUNS, ALL_SUBJECTS, DEFAULT_SPACE  # noqa: E402
from emfl.glm import EFMLOCFirstLevelGLM  # noqa: E402

MODALITIES = ("visual", "auditory")


def build_analyzer(derivatives_root, subject_id: str, space: str = DEFAULT_SPACE,
                   smoothing_fwhm: float = 3.0, run_split: str = "all",
                   raw_root=None) -> EFMLOCFirstLevelGLM:
    """Construct the vendored GLM engine with an explicit events (raw BIDS) root.

    ``raw_root`` is threaded to the engine's ``orig_data_dir`` param (events live in raw
    BIDS, not derivatives). When None the engine falls back to its dev sibling-path hack.
    """
    orig_data_dir = str(Path(raw_root)) if raw_root else None
    return EFMLOCFirstLevelGLM(
        derivatives_dir=str(derivatives_root),
        subject_id=subject_id,
        space=space,
        smoothing_fwhm=smoothing_fwhm,
        run_split=run_split,
        orig_data_dir=orig_data_dir,
    )


def run_subject_run(analyzer: EFMLOCFirstLevelGLM, subject_id: str, run: str,
                    save_outputs: bool = True) -> dict:
    """Run visual + auditory GLM for one (subject, run) via an already-built analyzer.

    Returns a status record (mirrors dev `process_single_subject_run`). Side effect: the
    analyzer writes the contrast/effect maps to disk when ``save_outputs`` is True — that
    is the whole point of the raw-path step.
    """
    result = {
        "subject": subject_id,
        "run": run,
        "run_split": analyzer.run_split,
        "space": analyzer.space,
        "errors": [],
    }
    for modality in MODALITIES:
        try:
            out = analyzer.run_effloc_glm(run, modality=modality, save_outputs=save_outputs)
            if analyzer.is_surface:
                n_contrasts = len(out["L"][4])
            else:
                _glm, contrasts = out
                n_contrasts = len(contrasts)
            result[f"{modality}_status"] = "success"
            result[f"{modality}_contrasts"] = n_contrasts
        except Exception as e:  # noqa: BLE001 - match dev: record error, continue
            result[f"{modality}_status"] = "failed"
            result[f"{modality}_contrasts"] = 0
            result["errors"].append(f"{modality} run {run}: {e}")
    return result


def run_first_level(subjects, derivatives_root, raw_root=None, runs=None,
                    space: str = DEFAULT_SPACE, smoothing_fwhm: float = 3.0,
                    save_outputs: bool = True) -> list:
    """Per-run first-level GLM (run_split='all') for the given subjects. Returns status records."""
    runs = list(runs) if runs is not None else list(ALL_RUNS)
    results = []
    for subject_id in subjects:
        analyzer = build_analyzer(
            derivatives_root, subject_id, space=space, smoothing_fwhm=smoothing_fwhm,
            run_split="all", raw_root=raw_root)
        for run in runs:
            print(f"\n{'#'*70}\n# {subject_id} | run {run}\n{'#'*70}")
            results.append(run_subject_run(analyzer, subject_id, run, save_outputs=save_outputs))
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derivatives-root", required=True,
                   help="fMRIPrep derivatives root (holds <subj>/func/; GLM maps written "
                        "under <subj>/first_level_glm/).")
    p.add_argument("--raw-root", default=None,
                   help="Raw BIDS root for event TSVs (<subj>/func/*_events.tsv). If "
                        "omitted, the engine derives it from --derivatives-root (dev hack).")
    p.add_argument("--subjects", nargs="+", default=list(ALL_SUBJECTS),
                   help="Subject IDs (default: all 6 EMFL subjects).")
    p.add_argument("--runs", nargs="+", default=list(ALL_RUNS),
                   help="Run numbers (default: all 5 runs).")
    p.add_argument("--space", default=DEFAULT_SPACE,
                   choices=["MNI152NLin2009cAsym", "T1w", "fsnative", "fsaverage5", "fsaverage6"],
                   help="Analysis space (Branch A: MNI 2mm).")
    p.add_argument("--smoothing", type=float, default=3.0,
                   help="Smoothing FWHM in mm (default 3.0; matches Marvi et al. 2025).")
    p.add_argument("--no-save", action="store_true",
                   help="Do not write maps to disk (dry validation).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    print("=" * 70)
    print("FIRST-LEVEL EMFL GLM (per run) — Branch A raw-path step")
    print(f"  subjects   : {', '.join(args.subjects)}")
    print(f"  runs       : {', '.join(args.runs)}")
    print(f"  derivatives: {args.derivatives_root}")
    print(f"  raw (events): {args.raw_root or '(derived from derivatives)'}")
    print(f"  space      : {args.space}   smoothing: {args.smoothing}mm")
    print("=" * 70)

    results = run_first_level(
        subjects=args.subjects, derivatives_root=args.derivatives_root,
        raw_root=args.raw_root, runs=args.runs, space=args.space,
        smoothing_fwhm=args.smoothing, save_outputs=not args.no_save)

    n_ok = sum(1 for r in results
               if r.get("visual_status") == "success" and r.get("auditory_status") == "success")
    print(f"\n{n_ok}/{len(results)} subject-run pairs fully succeeded.")
    failed = [r for r in results if r["errors"]]
    for r in failed:
        print(f"  ! {r['subject']} run {r['run']}: {'; '.join(r['errors'])}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
