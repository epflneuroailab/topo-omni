"""Drift guard — the local engine constants must match the dataset config (docs/DESIGN.md §4).

`analysis/glm_engine.py` keeps its own copies of the 24 standard confounds and the 83
canonical subjects so the engine stays self-contained and relocatable into `topo-omni`
(mirrors Pernet, whose engine keeps its own fixed thresholds). This test prevents the
two copies from silently diverging. Pure-Python; no data required.
"""
import importlib.util
from pathlib import Path

_DATASET = Path(__file__).resolve().parent.parent


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _DATASET / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


config = _load("jung_config", "config.py")


def _load_engine():
    import sys
    sys.path.insert(0, str(_DATASET / "analysis"))
    return _load("jung_glm_engine", "analysis/glm_engine.py")


def test_confounds_match():
    engine = _load_engine()
    assert engine.STANDARD_CONFOUNDS == config.STANDARD_CONFOUNDS
    assert engine.N_CONFOUNDS == config.N_CONFOUNDS == 24


def test_canonical_subjects_match():
    engine = _load_engine()
    assert engine.CANONICAL_SUBJECTS == config.CANONICAL_SUBJECTS
    assert engine.N_SUBJECTS == 83


def test_tr_matches():
    engine = _load_engine()
    assert engine.TR == config.TR == 0.46


def test_dropped_subjects_are_canonical():
    # The 5 dropped subjects must actually be in the canonical list.
    for s in config.CONFOUND_DROPPED_SUBJECTS:
        assert f"sub-{s}" in config.CANONICAL_SUBJECTS
