"""Characterization test — the published n=78 / df=77 subject drop (docs/DESIGN.md §7, index §8.2).

This is a *reproduction* pin, not a bug: the confound loader hard-requires all 24 named
confounds in every run, and fMRIPrep legitimately emits fewer `cosineNN` (run-length
dependent) or `t_comp_cor_NN` (variance dependent) columns for some runs. Five of the 83
canonical subjects therefore raise in `load_confounds` and are dropped by
`run_subject_analysis`'s blanket except, yielding the n=78 group maps in the paper. We
KEEP this behavior (correct only the paper text df=82→77); we do NOT pad to n=83
(no `--confounds-tolerant` deviation is built — docs/DESIGN.md §10).

The test documents *which* subjects drop and *why* (which confound column is missing).

DATA-GATED (needs the fMRIPrep confound TSVs): the full-83 scan is header-only (fast);
the loader-exercise runs the real ported `load_confounds` on the 5 dropped + 3 kept
controls. Skips cleanly when derivatives are absent. Point with JUNG_DERIVATIVES_ROOT.
"""
import importlib.util
import os
from pathlib import Path

import pytest

_DATASET = Path(__file__).resolve().parent.parent
_CANDIDATE_ROOTS = [
    os.environ.get("JUNG_DERIVATIVES_ROOT"),
    "/work/upschrimpf1/mehrer/datasets/fMRI_movie_watching/spacetop/ds005256/derivatives",
]

EXPECTED_DROPPED = {"sub-0035", "sub-0044", "sub-0061", "sub-0084", "sub-0131"}
# Why each drops (documented from the on-disk TSVs; index §8.2):
#   0035/0061/0084 — a run with fewer than 4 `cosine` columns (short run)
#   0044/0131      — a run missing `t_comp_cor_02` (low-variance tCompCor)


def _has_deps():
    try:
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_root():
    for root in _CANDIDATE_ROOTS:
        if root and (Path(root) / "sub-0001").exists():
            return Path(root)
    return None


def _load_engine():
    path = _DATASET / "analysis" / "glm_engine.py"
    spec = importlib.util.spec_from_file_location("jung_glm_engine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _subject_has_all_confounds(engine, subject_id, root):
    """Header-only mirror of load_confounds's raise condition (fast full-83 scan).

    Returns True iff every run has all 24 named confounds (with the same fuzzy match
    load_confounds uses). A subject that returns False is one load_confounds would drop.
    """
    import pandas as pd

    conf_files = sorted(Path(root).glob(
        f"{subject_id}/ses-*/func/*_desc-confounds_timeseries.tsv"
    ))
    if not conf_files:
        return False
    for f in conf_files:
        cols = list(pd.read_csv(f, sep="\t", nrows=0).columns)
        norm = {c.replace("_", "").lower() for c in cols}
        for confound in engine.STANDARD_CONFOUNDS:
            key = confound.replace("_", "").lower()
            if confound in cols:
                continue
            if not any(key in c for c in norm):
                return False
    return True


pytestmark = pytest.mark.skipif(not _has_deps(), reason="pandas not installed")


@pytest.fixture(scope="module")
def engine():
    return _load_engine()


def test_full_83_scan_yields_exactly_n78(engine):
    """All 83 canonical subjects → exactly 78 retained, exactly the 5 named dropped."""
    root = _resolve_root()
    if root is None:
        pytest.skip("fMRIPrep confound TSVs not found (set JUNG_DERIVATIVES_ROOT)")

    dropped = {s for s in engine.CANONICAL_SUBJECTS
               if not _subject_has_all_confounds(engine, s, root)}

    assert dropped == EXPECTED_DROPPED, f"drop set changed: {sorted(dropped)}"
    assert len(engine.CANONICAL_SUBJECTS) - len(dropped) == 78


@pytest.mark.parametrize("subject", sorted(EXPECTED_DROPPED))
def test_loader_actually_raises_on_dropped(engine, subject):
    """The real ported load_confounds raises on each of the 5 dropped subjects."""
    root = _resolve_root()
    if root is None:
        pytest.skip("fMRIPrep confound TSVs not found (set JUNG_DERIVATIVES_ROOT)")
    with pytest.raises((ValueError, FileNotFoundError)):
        engine.load_confounds(subject, str(root))


@pytest.mark.parametrize("subject", ["sub-0001", "sub-0002", "sub-0133"])
def test_loader_keeps_controls(engine, subject):
    """The real ported load_confounds returns a (n_trs, 24) matrix for kept subjects."""
    root = _resolve_root()
    if root is None:
        pytest.skip("fMRIPrep confound TSVs not found (set JUNG_DERIVATIVES_ROOT)")
    confounds = engine.load_confounds(subject, str(root))
    assert confounds.shape[1] == 24
