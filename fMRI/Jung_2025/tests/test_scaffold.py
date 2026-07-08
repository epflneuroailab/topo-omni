"""Jung scaffold tests — parser wiring, dispatch contract, and the n=78 pin (docs/DESIGN.md §6/§7).

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


config = _load("config", "config.py")
make_figures = _load("jung_make_figures", "make_figures.py")


def test_parser_defaults_to_precomputed():
    args = make_figures.build_parser().parse_args([])
    assert args.input_source == "precomputed"
    assert args.figures == list(make_figures.FIGURES)


def test_jung_has_derivatives_root():
    dests = {a.dest for a in make_figures.build_parser()._actions}
    assert "derivatives_root" in dests


def test_n78_reproduction_pin_is_recorded():
    # The published analysis is n=78 / df=77 with exactly these 5 subjects dropped
    # (docs/DESIGN.md §7). Pinned here at scaffold stage; the characterization test on the
    # real confound loader (added during the port) enforces the actual drop.
    assert config.N_SUBJECTS_PUBLISHED == 78
    assert config.DF_PUBLISHED == 77
    assert set(config.CONFOUND_DROPPED_SUBJECTS) == {"0035", "0044", "0061", "0084", "0131"}
    assert config.RAW_SOURCE["n_subjects_available"] - len(config.CONFOUND_DROPPED_SUBJECTS) == config.N_SUBJECTS_PUBLISHED


def test_figures_are_the_two_paper_branches():
    # Both figures draw from the single new54 family (App. D "14 clusters" = typo for 54).
    assert make_figures.FIGURES == ("fig6_d4", "figD5")
    assert make_figures._FIG_SPEC["fig6_d4"] == ("new54", "54_cluster.json", [5, 32, 49])
    assert make_figures._FIG_SPEC["figD5"] == ("new54", "54_cluster.json", [6, 30, 31])
    # config is the single source of truth for the figure → cluster-ID mapping.
    assert config.FIGURES["fig6_d4"]["ids"] == [5, 32, 49]
    assert config.FIGURES["figD5"]["ids"] == [6, 30, 31]
    assert list(config.CLUSTERS) == ["new54"]


def test_plots_default_to_dataset_plots_dir():
    # Figures default into <dataset>/plots, decoupled from --derivatives-root/--results-root.
    from pathlib import Path
    args = make_figures.build_parser().parse_args([])
    assert Path(args.plots_root).name == "plots"
    assert Path(args.plots_root).parent.name == "Jung_2025"


def test_dispatch_requires_derivatives_root():
    # The GLM cannot run without the precomputed fsaverage6 cut; the driver guards it.
    with pytest.raises(SystemExit):
        make_figures.main(["--input-source", "precomputed"])
