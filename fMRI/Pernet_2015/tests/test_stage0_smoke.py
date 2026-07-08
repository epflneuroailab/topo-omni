"""Pernet Stage-0 smoke tests (faithful port; not golden-mastered — docs/DESIGN.md §2.5/§6).

Stage 0 (raw BIDS -> precomputed cut) is env-pinned (nilearn 0.10.4) + FSL-dependent, so
it can't run in CI. These tests cover the wiring instead:

  - preprocessing/run_stage0.py is import-light (nibabel/nilearn/FSL are imported lazily
    inside its run functions), so the CLI / SLURM-emitter surface is exercised with no
    heavy deps — runs in the fast suite.
  - The full package (relative imports + nibabel/nilearn) is imported in an isolated
    subprocess run from the dataset dir, gated on those deps being present.

The make_figures `--input-source raw` dispatch/validation smoke is in test_scaffold.py.
"""
import importlib.util
import os
import subprocess
import sys

import pytest

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Pernet_2015/


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_DIR, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# run_stage0's top level is dependency-light — load it by path (no package context needed;
# its relative imports live inside the run functions we don't call here).
stage0 = _load("pernet_run_stage0", "preprocessing/run_stage0.py")


def test_subject_list_matches_template():
    assert stage0.subject_list(3) == ["sub001_Ed", "sub002_Ed", "sub003_Ed"]
    assert len(stage0.subject_list()) == 218
    assert stage0.N_SUBJECTS == 218


def test_cli_requires_step_and_roots():
    with pytest.raises(SystemExit):
        stage0.build_parser().parse_args([])  # missing positional step + required roots
    args = stage0.build_parser().parse_args(
        ["glm", "--raw-root", "/r", "--results-root", "/o"]
    )
    assert args.step == "glm" and args.raw_root == "/r" and args.results_root == "/o"


def test_slurm_emitter_shape():
    script = stage0.emit_slurm_script("cv-split", "/raw", "/out", n_subjects=5)
    assert "#SBATCH --array=1-5" in script
    assert "run_stage0.py cv-split" in script
    assert '"sub001_Ed"' in script and '"sub005_Ed"' in script
    # Site-specific env/FSL activation is a marked placeholder, not hard-coded.
    assert "SITE-SPECIFIC" in script


def test_figure_to_step_mapping():
    # Fig. 3b map + Fig. B3b need the volumetric GLM cut; the 2-bar profile needs the CV cut.
    assert stage0._NEEDS_GLM == {"fig3b_map", "figB3b_morans_i"}
    assert stage0._NEEDS_CV == {"fig3b_profile"}


_HAVE_HEAVY = all(importlib.util.find_spec(m) is not None for m in ("nibabel", "nilearn"))


@pytest.mark.skipif(
    not _HAVE_HEAVY,
    reason="Stage-0 package needs nibabel + nilearn (the pinned analysis env)",
)
def test_preprocessing_package_imports_and_path_guard():
    # Isolated subprocess, run from the dataset dir so `import preprocessing` is unambiguous
    # (all three datasets ship a `preprocessing` package). Confirms the modules import and
    # the path-agnostic guard (no baked-in dataset path) fires.
    prog = (
        "import preprocessing.run_stage0, preprocessing.cv_split, preprocessing.volumetric_glm\n"
        "import preprocessing.motion_correction, preprocessing.timing\n"
        "from preprocessing.data_loader import Pernet2015DataLoader\n"
        "try:\n"
        "    Pernet2015DataLoader(base_path=None)\n"
        "    raise SystemExit('guard did not raise')\n"
        "except ValueError:\n"
        "    pass\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", prog], cwd=_DIR, capture_output=True, text=True
    )
    assert r.returncode == 0, f"stdout={r.stdout!r}\nstderr={r.stderr!r}"
