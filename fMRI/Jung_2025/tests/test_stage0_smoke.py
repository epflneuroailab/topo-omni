"""Stage-0 smoke tests — raw-path dispatch + preprocessing artifacts (docs/DESIGN.md §6).

Stage 0 (fMRIPrep) is containerized and NOT golden-mastered. The default reproduction
runs `--input-source precomputed`, so the `raw` lineage is otherwise never exercised and
would bit-rot. These cheap tests assert the raw entry point parses/dispatches and the
ported Stage-0 scripts are present and shaped correctly. No data, no fMRIPrep required.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_DATASET = Path(__file__).resolve().parent.parent
_PREP = _DATASET / "preprocessing"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _DATASET / filename)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# make_figures does `import config` at top level → register config under that name first.
_load("config", "config.py")
make_figures = _load("jung_make_figures_stage0", "make_figures.py")


def test_raw_source_parses_and_dispatches():
    args = make_figures.build_parser().parse_args(
        ["--input-source", "raw", "--raw-root", "/x", "--derivatives-root", "/y"]
    )
    assert args.input_source == "raw"
    assert args.raw_root == "/x"
    assert args.derivatives_root == "/y"


def test_raw_requires_raw_root_for_csv_regen():
    # The raw path regenerates the cluster CSV from events → needs --raw-root.
    with pytest.raises(SystemExit):
        make_figures.main(["--input-source", "raw", "--derivatives-root", "/y",
                           "--figures", "fig6_d4", "--skip-visualize"])


def test_stage0_scripts_present():
    for name in ("download_raw_data.sh", "submit_fmriprep.sh",
                 "fmriprep_single_subject.sbatch", "bids_filter_alignvideo.json"):
        assert (_PREP / name).exists(), f"missing Stage-0 file: {name}"


def test_bids_filter_restricts_to_alignvideo():
    filt = json.loads((_PREP / "bids_filter_alignvideo.json").read_text())
    assert filt["bold"]["task"] == "alignvideo"
    assert filt["sbref"]["task"] == "alignvideo"


def test_fmriprep_sbatch_targets_fsaverage6_surface_only():
    txt = (_PREP / "fmriprep_single_subject.sbatch").read_text()
    assert "fsaverage6" in txt
    # MNI/fsLR are intentionally skipped (surface-only); the flag lists only surface spaces.
    assert "--output-spaces fsaverage5 fsaverage6 anat" in txt
    assert "fmriprep-24.0.1" in txt  # pinned container version documented
