"""Marvi scaffold tests — parser wiring + dispatch contract (docs/DESIGN.md §6).

See Pernet_2015/tests/test_scaffold.py for why modules are loaded by explicit path.
"""
import importlib.util
import os
import sys

import pytest

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(unique_name, filename):
    spec = importlib.util.spec_from_file_location(unique_name, os.path.join(_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


_config = _load("config", "config.py")
make_figures = _load("marvi_make_figures", "make_figures.py")


def test_parser_defaults_to_precomputed():
    args = make_figures.build_parser().parse_args([])
    assert args.input_source == "precomputed"
    assert args.figures == list(make_figures.FIGURES)


def test_marvi_has_derivatives_root():
    # Marvi's cut is fMRIPrep derivatives -> --derivatives-root (docs/DESIGN.md §1/§2.3).
    dests = {a.dest for a in make_figures.build_parser()._actions}
    assert "derivatives_root" in dests


def test_plots_default_to_dataset_plots_dir():
    # Figures default into <dataset>/plots, decoupled from --derivatives-root/--results-root.
    from pathlib import Path
    args = make_figures.build_parser().parse_args([])
    assert Path(args.plots_root).name == "plots"
    assert Path(args.plots_root).parent.name == "Marvi_2025"


def test_dispatch_requires_derivatives_root():
    # Both branches are now wired (figA2 + fig2/fig3); the dispatch is no longer a scaffold.
    # A precomputed run with no --derivatives-root must exit cleanly (needs the cut location),
    # not raise an unhandled error — the render handlers guard on it.
    with pytest.raises(SystemExit):
        make_figures.main(["--input-source", "precomputed"])
