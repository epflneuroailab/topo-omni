"""Branch B (Figs 2 & 3 native-surface) wiring smoke test — docs/DESIGN.md §6 Stage-0-style.

The native-surface render chain is exercised only for WIRING (no compute, no dataset):
  * `make_figures.render_fig2_fig3` orchestrates 09 project -> 11 inflated -> [12 parcels]
    -> 10 render in order, threading --derivatives-root / --results-root / --subjects, and
    renders with the PAPER metric (signed_log_p);
  * step 12 (FreeSurfer-CLI parcels) is only run when a container is present AND contours are
    not already shipped; otherwise the shipped contours are reused, or (neither) it renders
    with --no-contours;
  * the render module defaults to `signed_log_p` (paper) and its output-dir naming is stable;
  * the concat-GLM (08) parser carries parameterized roots and its contrasts match config.

DATA-GATED on nilearn (render module import) + numpy. Skips cleanly otherwise.
"""
import importlib
import sys
from pathlib import Path

import pytest

_DATASET = Path(__file__).resolve().parent.parent
if str(_DATASET) not in sys.path:
    sys.path.insert(0, str(_DATASET))


def _has_deps():
    try:
        import nilearn  # noqa: F401
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_deps(), reason="nilearn + numpy (Stage-1 stack) not installed")


def _patch_chain(monkeypatch, mf):
    """Monkeypatch the four Branch-B analysis mains to order recorders; return the order list."""
    from analysis import (project_to_native_surface, convert_inflated_surfaces,
                          project_parcels_to_surface, visualize_native_surface)
    order = []

    def rec(tag, ret=0):
        def _f(argv=None, *a, **k):
            order.append((tag, list(argv) if argv else []))
            return ret
        return _f

    monkeypatch.setattr(project_to_native_surface, "main", rec("project09"))
    monkeypatch.setattr(convert_inflated_surfaces, "main", rec("inflated11"))
    monkeypatch.setattr(project_parcels_to_surface, "main", rec("parcels12"))
    monkeypatch.setattr(visualize_native_surface, "main", rec("render10"))
    return order, project_parcels_to_surface


def test_branch_b_mains_accept_argv():
    """Every Branch-B analysis main MUST accept an argv list — make_figures calls e.g.
    `project_to_native_surface.main([...])`. The order-recorder tests above monkeypatch the
    mains, so they never exercise the real signatures; this guards the actual functions.
    (Regression: 09/11/12 shipped `def main():` — argv-less — so make_figures Branch B raised
    `TypeError: main() takes 0 positional arguments` end-to-end; caught by the step-2 cut gate.)
    """
    import inspect
    from analysis import (project_to_native_surface, convert_inflated_surfaces,
                          project_parcels_to_surface, visualize_native_surface)
    for mod in (project_to_native_surface, convert_inflated_surfaces,
                project_parcels_to_surface, visualize_native_surface):
        params = inspect.signature(mod.main).parameters
        positional = [p for p in params.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        assert positional and positional[0].name == "argv", (
            f"{mod.__name__}.main must accept argv as its first parameter (got {list(params)})")
        # and it must be optional (default None) so `if __name__` main() still works
        assert positional[0].default is None, f"{mod.__name__}.main argv must default to None"


def test_render_fig2_fig3_uses_shipped_contours(monkeypatch, tmp_path):
    """When native_surface_parcels/ is shipped, 12 is NOT run and contours stay ON."""
    mf = importlib.import_module("make_figures")
    order, pps = _patch_chain(monkeypatch, mf)
    (tmp_path / "native_surface_parcels").mkdir()  # shipped contours present

    args = mf.build_parser().parse_args([
        "--derivatives-root", str(tmp_path), "--results-root", str(tmp_path / "out"),
        "--plots-root", str(tmp_path / "plots"),
        "--subjects", "sub-kaneff01", "--figures", "fig2_surface"])
    mf.render_fig2_fig3(args)

    tags = [t for t, _ in order]
    assert tags == ["project09", "inflated11", "render10"], tags  # 12 skipped
    steps = dict(order)
    # roots + subjects threaded
    assert str(tmp_path) in steps["project09"] and "sub-kaneff01" in steps["project09"]
    assert str(tmp_path) in steps["inflated11"]
    # paper metric, contours ON (no --no-contours)
    assert "signed_log_p" in steps["render10"]
    assert "--no-contours" not in steps["render10"]
    # figures now default to the plots dir, not the results/derivatives cut
    assert str(tmp_path / "plots") in steps["render10"]


def test_render_fig2_fig3_runs_step12_when_container(monkeypatch, tmp_path):
    """No shipped contours + FreeSurfer container available -> step 12 runs."""
    mf = importlib.import_module("make_figures")
    order, pps = _patch_chain(monkeypatch, mf)
    monkeypatch.setattr(pps, "_freesurfer_tools_available", lambda: True)

    args = mf.build_parser().parse_args([
        "--derivatives-root", str(tmp_path), "--plots-root", str(tmp_path / "plots"),
        "--subjects", "sub-kaneff01", "--figures", "fig3_surface"])
    mf.render_fig2_fig3(args)

    tags = [t for t, _ in order]
    assert tags == ["project09", "inflated11", "parcels12", "render10"], tags
    assert "--no-contours" not in dict(order)["render10"]


def test_render_fig2_fig3_no_contours_fallback(monkeypatch, tmp_path):
    """No shipped contours + no container -> render WITHOUT contours (still 09/11/10)."""
    mf = importlib.import_module("make_figures")
    order, pps = _patch_chain(monkeypatch, mf)
    monkeypatch.setattr(pps, "_freesurfer_tools_available", lambda: False)

    args = mf.build_parser().parse_args([
        "--derivatives-root", str(tmp_path), "--plots-root", str(tmp_path / "plots"),
        "--subjects", "sub-kaneff01", "--figures", "fig2_surface"])
    mf.render_fig2_fig3(args)

    tags = [t for t, _ in order]
    assert tags == ["project09", "inflated11", "render10"], tags  # 12 skipped (no container)
    assert "--no-contours" in dict(order)["render10"]


def test_render_fig2_fig3_requires_derivatives_root():
    mf = importlib.import_module("make_figures")
    args = mf.build_parser().parse_args(["--figures", "fig2_surface"])  # no --derivatives-root
    with pytest.raises(SystemExit):
        mf.render_fig2_fig3(args)


def test_visualize_native_surface_defaults_to_paper_metric():
    vns = importlib.import_module("analysis.visualize_native_surface")
    # Real parser defaults (required roots supplied so parse succeeds).
    args = vns.build_parser().parse_args(["--derivatives-root", "/d", "--output-dir", "/o"])
    assert args.metric == "signed_log_p", "release render must default to the paper metric"
    assert args.threshold == 3.0
    # t_fdr (dev exploratory) remains available.
    alt = vns.build_parser().parse_args(
        ["--derivatives-root", "/d", "--output-dir", "/o", "--metric", "t_fdr"])
    assert alt.metric == "t_fdr"
    # output-dir naming is stable + encodes the signed-log-p p<0.001 threshold
    root = vns.output_root(Path("/out"), "signed_log_p", 3.0, 0.05, no_contours=False)
    assert root.name == "subject_level_native_surface_T1w_with_anat_contours_signed_log_p_p0.001_thresh3.0"
    root_nc = vns.output_root(Path("/out"), "signed_log_p", 3.0, 0.05, no_contours=True)
    assert "without_anat_contours" in root_nc.name


def test_concatenated_glm_parser_and_contrasts():
    cg = importlib.import_module("analysis.concatenated_glm")
    # contrast KEYS match config (single source of truth)
    import config
    vis = set(cg.concat_contrast_formulas("visual"))
    aud = set(cg.concat_contrast_formulas("auditory"))
    assert vis == set(config.VISUAL_CONTRASTS), vis
    assert aud == set(config.AUDITORY_CONTRASTS), aud
    # a couple of formulas are byte-faithful to dev 08
    assert cg.concat_contrast_formulas("visual")["faces_vs_objects"] == "faces - objects"
    assert cg.concat_contrast_formulas("auditory")["english_vs_nonwords"] == \
        "0.5*false_belief + 0.5*false_photo - nonwords"
