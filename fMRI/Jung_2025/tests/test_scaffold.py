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


def test_figure_is_the_revised_paper_fig6():
    # The revised paper's Fig. 6 = new16 clusters 7 (TV news interviews) and 10 (natural
    # landscapes), broken runs excluded. config is the single source of truth.
    assert make_figures.FIGURES == ("fig6",)
    assert make_figures._FIG_SPEC["fig6"] == ("new16", "16_cluster.json", [7, 10], True)
    assert config.FIGURES["fig6"]["ids"] == [7, 10]
    assert config.FIGURES["fig6"]["family"] == "new16"
    assert config.FIGURES["fig6"]["exclude_broken_runs"] is True
    # new54 (arXiv v1/v2 maps) and new27 stay registered as families.
    assert list(config.CLUSTERS) == ["new54", "new27", "new16"]


def test_fig6_dispatch_excludes_broken_runs(monkeypatch, tmp_path):
    # The precomputed path must call the contrast driver with --exclude-broken-runs and read /
    # write the *_exclbrokenruns subdir (the shipped OSF cut layout).
    calls = []

    class _FakeContrasts:
        FAMILIES = {"new16": {"derivatives_subdir": "cluster_contrasts_new16clusters"}}

        @staticmethod
        def main(argv):
            calls.append(list(argv))
            return 0

    monkeypatch.setattr(make_figures, "_load", lambda name: _FakeContrasts)
    make_figures.main(["--derivatives-root", str(tmp_path), "--skip-visualize"])
    assert [c[c.index("--cluster-id") + 1] for c in calls] == ["7", "10"]
    sub = "cluster_contrasts_new16clusters_exclbrokenruns"
    for argv in calls:
        assert "--exclude-broken-runs" in argv
        assert argv[argv.index("--family") + 1] == "new16"
        assert argv[argv.index("--output-dir") + 1] == str(tmp_path / sub)
        assert argv[argv.index("--subject-maps-root") + 1] == str(tmp_path / sub)


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
