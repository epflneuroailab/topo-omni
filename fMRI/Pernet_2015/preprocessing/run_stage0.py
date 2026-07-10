"""Pernet Stage-0 orchestrator: raw BIDS -> the precomputed cut.

Regenerates what the `precomputed` path ships, from the raw Edinburgh DataShare
dataset (FSL preprocessing + Nilearn first-level GLM). This is the `--input-source
raw` back-end for make_figures.py; the two paths converge at the on-disk contrast maps.

Index mapping (20241003_pernet_2015 @ f842b1a; docs/DESIGN.md §9):
  step "glm"        = 00_volumetric_glm_parallel.py + run_single_subject_glm.py
                      -> <results-root>/00_volumetric_GLM/sub*/sub*_contrast_estimates.nii.gz
                         (needed by Fig. 3b map and Fig. B3b)
  step "fold-split" = cv_01_define_fold_split.py
                      -> <results-root>/04_cross_validation/fold_split.json
  step "cv-split"   = cv_02_split_glm_single_subject.py (submitted by cv_03 as a SLURM array)
                      -> <results-root>/04_cross_validation/per_subject/sub*/half-*.nii.gz
                         (needed by Fig. 3b 2-bar profile)

Path-agnostic: raw BIDS root (`--raw-root`) and output root (`--results-root`) are
required arguments — no baked-in dataset path. FSL binaries must already be on PATH
(FSLDIR set); the dev repo sourced setup_fsl.sh, which is site-specific and not shipped.

Stage 0 is env-pinned (nilearn 0.10.4) + FSL-dependent, so it is NOT golden-mastered:
faithful port + raw-dispatch smoke test (docs/DESIGN.md §2.5/§6). Heavy imports
(nibabel/nilearn/FSL) are performed lazily inside the run functions, so importing this
module (for its CLI / SLURM emitter) does not require them.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Stage-0 facts (mirror Pernet_2015/config.py; kept here so the package is self-contained).
N_SUBJECTS = 218
SUBJECT_TEMPLATE = "sub{:03d}_Ed"
TVA_LOC_RELPATH = "voice_localizer/TVA_loc.txt"
SUBS_RELPATH = "subs"
GLM_SUBDIR = "00_volumetric_GLM"

# Which precomputed-cut artifact each figure lineage consumes (make_figures.FIGURES keys).
_NEEDS_GLM = {"fig3b_map", "figB3b_morans_i"}   # -> 00_volumetric_GLM/
_NEEDS_CV = {"fig3b_profile"}                    # -> 04_cross_validation/per_subject/


def subject_list(n_subjects: int = N_SUBJECTS) -> list[str]:
    return [SUBJECT_TEMPLATE.format(i) for i in range(1, n_subjects + 1)]


# ── step "glm" (00) ──────────────────────────────────────────────────────────
def run_glm_subject(subject_id: str, raw_root: str, results_root: str) -> None:
    """Volumetric first-level GLM for one subject -> 00_volumetric_GLM/<sub>/."""
    from .data_loader import Pernet2015DataLoader
    from .volumetric_glm import run_single_subject_volumetric_glm

    loader = Pernet2015DataLoader(base_path=str(raw_root))
    run_single_subject_volumetric_glm(
        subject_id,
        loader,
        output_dir=str(Path(results_root) / GLM_SUBDIR),
        tva_loc_path=str(Path(raw_root) / TVA_LOC_RELPATH),
        data_dir=str(Path(raw_root) / SUBS_RELPATH),
    )


# ── orchestrator used by make_figures --input-source raw ─────────────────────
def build_precomputed_cut(raw_root: str, results_root: str,
                          figures=None, n_subjects: int = N_SUBJECTS,
                          subjects=None) -> None:
    """Run only the Stage-0 steps the requested figures need, then return.

    make_figures then runs the identical Stage-1 handlers on <results-root>.
    """
    from .cv_split import define_fold_split, run_split_glm

    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"--raw-root not found: {raw_root}")

    figures = set(figures) if figures is not None else (_NEEDS_GLM | _NEEDS_CV)
    subjects = subjects or subject_list(n_subjects)
    need_glm = bool(figures & _NEEDS_GLM)
    need_cv = bool(figures & _NEEDS_CV)

    print(f"Stage 0: raw={raw_root} -> {results_root}")
    print(f"  figures={sorted(figures)}  subjects={len(subjects)}  "
          f"glm={need_glm} cv={need_cv}")

    if need_glm:
        print(f"\n[glm] volumetric first-level GLM for {len(subjects)} subjects "
              f"-> {GLM_SUBDIR}/")
        for i, subject_id in enumerate(subjects, 1):
            print(f"  ({i}/{len(subjects)}) {subject_id}")
            run_glm_subject(subject_id, str(raw_root), results_root)

    if need_cv:
        print("\n[fold-split] block half-split (seed=42) -> 04_cross_validation/fold_split.json")
        _, split_path = define_fold_split(str(raw_root / TVA_LOC_RELPATH), results_root)
        fold_split = _load_fold_split(results_root)
        print(f"  wrote {split_path}")
        print(f"\n[cv-split] per-subject half GLMs for {len(subjects)} subjects "
              f"-> 04_cross_validation/per_subject/")
        for i, subject_id in enumerate(subjects, 1):
            print(f"  ({i}/{len(subjects)}) {subject_id}")
            run_split_glm(subject_id, fold_split, str(raw_root), results_root)


def _load_fold_split(results_root: str) -> dict:
    from .cv_split import load_fold_split
    return load_fold_split(results_root)


# ── SLURM emitter (faithful to dev 00_volumetric_glm_parallel / cv_03) ───────
def emit_slurm_script(step: str, raw_root: str, results_root: str,
                      n_subjects: int = N_SUBJECTS,
                      mem: str = "80G", time: str = "4:00:00", cpus: int = 10) -> str:
    """Emit a portable SLURM array script that runs `step` per subject.

    Faithful in shape to the dev launchers, but SITE-SPECIFIC glue is left as
    clearly-marked placeholders (env activation, FSL setup) instead of the dev's
    hard-coded conda path + `source src/setup_fsl.sh`.
    """
    subjects = subject_list(n_subjects)
    subjects_str = " ".join(f'"{s}"' for s in subjects)
    this = "preprocessing/run_stage0.py"
    return f"""#!/bin/bash
#SBATCH --job-name=pernet_stage0_{step}
#SBATCH --output=logs/stage0_{step}_%A_%a.out
#SBATCH --error=logs/stage0_{step}_%A_%a.err
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --array=1-{len(subjects)}

# --- SITE-SPECIFIC (fill in): activate the pinned env (nilearn 0.10.4) and put FSL on PATH ---
# eval "$(conda shell.bash hook)" && conda activate <analysis_env_pernet>
# export FSLDIR=/path/to/fsl && source $FSLDIR/etc/fslconf/fsl.sh && export PATH=$FSLDIR/bin:$PATH
# --------------------------------------------------------------------------------------------

SUBJECTS=({subjects_str})
SUBJECT_ID="${{SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}}"
echo "Subject: $SUBJECT_ID  ($(date))"

python {this} {step} \\
    --raw-root "{raw_root}" \\
    --results-root "{results_root}" \\
    --subject-id "$SUBJECT_ID"
"""


# ── CLI ──────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("glm", "fold-split", "cv-split", "all"),
                   help="Which Stage-0 step to run (glm=00, fold-split=cv_01, cv-split=cv_02).")
    p.add_argument("--raw-root", required=True, help="Raw Pernet BIDS root (has subs/ and voice_localizer/).")
    p.add_argument("--results-root", required=True, help="Where the precomputed cut is written.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--subject-id", default=None, help="Process a single subject (e.g. sub001_Ed).")
    g.add_argument("--n-subjects", type=int, default=N_SUBJECTS, help="Process subjects 1..N (default 218).")
    p.add_argument("--slurm", action="store_true", help="Emit a SLURM array script for `step` instead of running.")
    p.add_argument("--dry-run", action="store_true", help="With --slurm, print the script to stdout.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_root, results_root = args.raw_root, args.results_root

    if args.slurm:
        step = "all" if args.step == "all" else args.step
        script = emit_slurm_script(step, raw_root, results_root, n_subjects=args.n_subjects)
        if args.dry_run:
            print(script)
        else:
            Path("logs").mkdir(exist_ok=True)
            out = Path(f"stage0_{step}.slurm")
            out.write_text(script)
            print(f"Wrote {out} (submit with: sbatch {out})")
        return 0

    subjects = [args.subject_id] if args.subject_id else subject_list(args.n_subjects)

    if args.step in ("glm", "all"):
        for i, s in enumerate(subjects, 1):
            print(f"[glm] ({i}/{len(subjects)}) {s}  ({datetime.now():%Y-%m-%d %H:%M:%S})")
            run_glm_subject(s, raw_root, results_root)

    if args.step in ("fold-split", "cv-split", "all"):
        from .cv_split import define_fold_split, run_split_glm
        if args.step in ("fold-split", "all"):
            _, path = define_fold_split(str(Path(raw_root) / TVA_LOC_RELPATH), results_root)
            print(f"[fold-split] wrote {path}")
        if args.step in ("cv-split", "all"):
            fold_split = _load_fold_split(results_root)
            for i, s in enumerate(subjects, 1):
                print(f"[cv-split] ({i}/{len(subjects)}) {s}")
                run_split_glm(s, fold_split, raw_root, results_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
