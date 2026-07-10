"""Pernet scaffold tests — parser wiring + dispatch contract.

Real tests that pass now; they guard the CLI surface (docs/DESIGN.md §2.3) and the scaffold
contract. Golden-master tests (docs/DESIGN.md §6 Tier 1) replace/extend these per lineage.

Runtime contract: each dataset is run from within its own folder, where `import
config` resolves to the local config.py. The three datasets share the bare module
names `make_figures`/`config`, so for a single root `pytest` run we load THIS
dataset's copies by explicit path — registering `config` in sys.modules before
loading `make_figures` (which imports it) so the binding is captured for good.
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


# Register config under the bare name `make_figures` imports, then load make_figures.
_config = _load("config", "config.py")
make_figures = _load("pernet_make_figures", "make_figures.py")


def test_parser_defaults_to_precomputed():
    args = make_figures.build_parser().parse_args([])
    assert args.input_source == "precomputed"
    assert args.figures == list(make_figures.FIGURES)


def test_pernet_has_results_root_not_derivatives_root():
    # Pernet's cut is contrast-level -> --results-root, no --derivatives-root (docs/DESIGN.md §1/§2.3).
    dests = {a.dest for a in make_figures.build_parser()._actions}
    assert "results_root" in dests
    assert "derivatives_root" not in dests


def test_precomputed_requires_results_root():
    with pytest.raises(SystemExit):
        make_figures.main(["--input-source", "precomputed"])


def test_raw_requires_raw_root():
    # raw is now wired (Stage-0 preprocessing) — with --results-root but no --raw-root it
    # must fail arg validation (SystemExit), not fall through or raise NotImplementedError.
    with pytest.raises(SystemExit):
        make_figures.main(["--input-source", "raw", "--results-root", "/tmp/pernet_out"])


def test_raw_validates_root_before_heavy_import():
    # A non-existent --raw-root is rejected during validation, before the Stage-0 package
    # (nibabel/nilearn/FSL) is imported — so this stays a fast, dependency-free check.
    with pytest.raises(SystemExit):
        make_figures.main([
            "--input-source", "raw",
            "--results-root", "/tmp/pernet_out",
            "--raw-root", "/nonexistent/pernet/raw",
        ])


def test_plots_default_to_dataset_plots_dir():
    # Figures default into <dataset>/plots, decoupled from the cut/--results-root.
    from pathlib import Path
    args = make_figures.build_parser().parse_args([])
    assert Path(args.plots_root).name == "plots"
    assert Path(args.plots_root).parent.name == "Pernet_2015"


def test_wired_figures_are_registered():
    # All three Stage-1 figure lineages are ported and dispatchable.
    assert set(make_figures.DISPATCH) == {"fig3b_map", "fig3b_profile", "figB3b_morans_i"}
    assert set(make_figures.DISPATCH) == set(make_figures.FIGURES)
