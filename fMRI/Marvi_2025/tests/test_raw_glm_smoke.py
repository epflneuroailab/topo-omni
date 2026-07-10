"""Raw-path dispatch smoke test — Branch A first-level GLM (docs/DESIGN.md §6 Stage-0-style).

The heavy nilearn first-level GLM is **NOT** golden-mastered (not bitwise-reproducible; the
published cut is a historical accretion across engine versions — see README). Per
docs/DESIGN.md §6, the deliverable for a raw/Stage-0-style step is a faithful port + a smoke test
that the `--input-source raw` entry points **parse and dispatch** — otherwise the raw
lineage (never exercised by the default precomputed reproduction) silently bit-rots.

This test therefore exercises only wiring, no GLM compute and no dataset:
  * both GLM drivers' argparse parsers build and carry the parameterized roots;
  * the vendored engine exposes the new explicit-events `orig_data_dir` param, and
    `first_level_glm.build_analyzer` threads `--raw-root` into it;
  * `glm_splits` reuses the shared engine wrappers and honours RUN_SPLITS;
  * `make_figures --input-source raw` validates the required roots and dispatches Branch A
    to the split-GLM producer (monkeypatched to a no-op recorder — zero compute).

DATA-GATED on nilearn only (importing the GLM drivers imports `emfl.glm` -> nilearn). Skips
cleanly when the Stage-1 stack is absent. No derivatives / raw BIDS required.
"""
import importlib
import inspect
import sys
from pathlib import Path

import pytest

_DATASET = Path(__file__).resolve().parent.parent
if str(_DATASET) not in sys.path:
    sys.path.insert(0, str(_DATASET))


def _has_nilearn():
    try:
        import nilearn  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_nilearn(), reason="nilearn (Stage-1 stack) not installed")


def test_first_level_parser_carries_parameterized_roots():
    fl = importlib.import_module("analysis.first_level_glm")
    args = fl.build_parser().parse_args([
        "--derivatives-root", "/d", "--raw-root", "/r",
        "--subjects", "sub-kaneff01", "sub-kaneff06", "--runs", "001", "003", "--no-save"])
    assert args.derivatives_root == "/d"
    assert args.raw_root == "/r"
    assert args.subjects == ["sub-kaneff01", "sub-kaneff06"]
    assert args.runs == ["001", "003"]
    assert args.no_save is True
    # --derivatives-root is required.
    with pytest.raises(SystemExit):
        fl.build_parser().parse_args(["--raw-root", "/r"])


def test_splits_parser_and_choices():
    gs = importlib.import_module("analysis.glm_splits")
    args = gs.build_parser().parse_args(["--derivatives-root", "/d", "--splits", "even", "odd"])
    assert args.splits == ["even", "odd"]
    # Unknown split is rejected by argparse choices.
    with pytest.raises(SystemExit):
        gs.build_parser().parse_args(["--derivatives-root", "/d", "--splits", "middle"])


def test_engine_exposes_explicit_events_root_and_zmap_helper():
    from emfl.glm import EFMLOCFirstLevelGLM
    params = inspect.signature(EFMLOCFirstLevelGLM.__init__).parameters
    assert "orig_data_dir" in params, "engine must accept an explicit events/raw root (PLAN §7)"
    # zmap-restore: the volumetric path must be able to emit z-score contrasts.
    assert hasattr(EFMLOCFirstLevelGLM, "_compute_zscore_contrasts")
    names = set(EFMLOCFirstLevelGLM._volumetric_contrast_formulas("visual"))
    assert names == {
        "faces_vs_objects", "scenes_vs_objects", "bodies_vs_objects",
        "words_vs_objects", "objects_vs_words"}, names


def test_build_analyzer_threads_raw_root_to_engine(monkeypatch):
    """build_analyzer must pass --raw-root through as the engine's orig_data_dir."""
    fl = importlib.import_module("analysis.first_level_glm")
    captured = {}

    class _FakeEngine:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(fl, "EFMLOCFirstLevelGLM", _FakeEngine)
    fl.build_analyzer("/deriv", "sub-kaneff01", raw_root="/raw", run_split="even")
    assert captured["orig_data_dir"] == "/raw"
    assert captured["derivatives_dir"] == "/deriv"
    assert captured["subject_id"] == "sub-kaneff01"
    assert captured["run_split"] == "even"
    # When raw_root is omitted, orig_data_dir is None -> engine falls back to the dev hack.
    captured.clear()
    fl.build_analyzer("/deriv", "sub-kaneff01")
    assert captured["orig_data_dir"] is None


def test_splits_reuses_shared_engine_wrappers():
    """glm_splits must delegate to first_level_glm's shared wrappers (no duplicated path logic)."""
    fl = importlib.import_module("analysis.first_level_glm")
    gs = importlib.import_module("analysis.glm_splits")
    assert gs.build_analyzer is fl.build_analyzer
    assert gs.run_subject_run is fl.run_subject_run


def test_make_figures_raw_requires_both_roots():
    mf = importlib.import_module("make_figures")
    # raw without --raw-root -> SystemExit (needs raw BIDS events).
    with pytest.raises(SystemExit):
        mf.main(["--input-source", "raw", "--derivatives-root", "/d", "--figures", "figA2_froi_profiles"])
    # raw without --derivatives-root -> SystemExit (needs a place to write the cut).
    with pytest.raises(SystemExit):
        mf.main(["--input-source", "raw", "--raw-root", "/r", "--figures", "figA2_froi_profiles"])


def test_make_figures_raw_dispatches_branch_a_to_split_glm(monkeypatch):
    """--input-source raw + figA2 must call glm_splits.run_glm_splits with the even/odd cut."""
    mf = importlib.import_module("make_figures")
    from analysis import glm_splits
    calls = {}

    def _fake_run_glm_splits(**kw):
        calls.update(kw)
        return []

    monkeypatch.setattr(glm_splits, "run_glm_splits", _fake_run_glm_splits)
    # Isolate the raw->GLM dispatch from the (separately tested) render chain.
    monkeypatch.setattr(mf, "render_figures", lambda args: 0)
    mf.main([
        "--input-source", "raw", "--derivatives-root", "/d", "--raw-root", "/r",
        "--subjects", "sub-kaneff01", "--figures", "figA2_froi_profiles"])
    assert calls, "raw figA2 did not dispatch to glm_splits.run_glm_splits"
    assert tuple(calls["splits"]) == ("even", "odd")
    assert calls["subjects"] == ["sub-kaneff01"]
    assert calls["raw_root"] == "/r"
    assert calls["derivatives_root"] == "/d"


def test_make_figures_render_chain_order_and_args(monkeypatch, tmp_path):
    """render_fig_a2 must call frois -> CV -> extract -> plot in order, threading roots/subjects."""
    from pathlib import Path
    mf = importlib.import_module("make_figures")
    from analysis import (cross_validation, define_frois,
                          extract_condition_responses, plot_figure_a2, plot_contrast_bars)
    order = []

    def rec(tag):
        def _f(argv=None, *a, **k):
            order.append((tag, list(argv) if argv else []))
            return 0
        return _f

    monkeypatch.setattr(define_frois, "main", rec("frois"))
    monkeypatch.setattr(cross_validation, "main", rec("cv"))
    monkeypatch.setattr(extract_condition_responses, "main", rec("extract"))
    monkeypatch.setattr(plot_figure_a2, "main", rec("plot"))
    monkeypatch.setattr(plot_contrast_bars, "main", rec("bars"))  # group contrast bars (M4)

    # render_fig_a2 creates --results-root (it may be fresh), so use writable tmp paths.
    deriv, out = str(tmp_path / "d"), str(tmp_path / "out")
    rc = mf.main([
        "--input-source", "precomputed", "--derivatives-root", deriv, "--results-root", out,
        "--subjects", "sub-kaneff01", "--figures", "figA2_froi_profiles"])
    assert rc == 0
    assert Path(out).is_dir()  # results-root was created
    assert [t for t, _ in order] == ["frois", "cv", "extract", "plot", "bars"], order
    steps = dict(order)
    # roots + subjects threaded through every step
    assert deriv in steps["frois"] and "sub-kaneff01" in steps["frois"]
    assert out in steps["extract"]  # extract writes the details CSV under --results-root
    # plot + group bars both read the details CSV that extract wrote
    assert any("condition_responses_details.csv" in x for x in steps["plot"])
    assert any("condition_responses_details.csv" in x for x in steps["bars"])


def test_make_figures_figa2_requires_derivatives_root():
    mf = importlib.import_module("make_figures")
    with pytest.raises(SystemExit):
        mf.main(["--figures", "figA2_froi_profiles"])  # no --derivatives-root


def test_make_figures_raw_branch_b_dispatches_to_concat_glm(monkeypatch):
    """--input-source raw + fig2/fig3 must regenerate the concat GLM (08), then render."""
    mf = importlib.import_module("make_figures")
    from analysis import concatenated_glm
    calls = {}

    def _fake_concat_main(argv=None, *a, **k):
        calls["argv"] = list(argv) if argv else []
        return 0

    monkeypatch.setattr(concatenated_glm, "main", _fake_concat_main)
    # Isolate the raw->GLM dispatch from the (separately tested) render chain.
    monkeypatch.setattr(mf, "render_figures", lambda args: 0)
    mf.main([
        "--input-source", "raw", "--derivatives-root", "/d", "--raw-root", "/r",
        "--subjects", "sub-kaneff01", "--figures", "fig2_surface"])
    assert calls, "raw fig2 did not dispatch to concatenated_glm.main"
    assert "/d" in calls["argv"] and "/r" in calls["argv"]
    assert "sub-kaneff01" in calls["argv"]
