#!/usr/bin/env python
"""Jung 2025 (Spacetop movie) — Stage 1 driver: precomputed cut -> paper figures.

Figure lineage (allow-list; docs/DESIGN.md §2.4). The revised paper's brain maps are two
SINGLE discovered clusters of the 16-cluster (new16) partition, with the four runs whose
events file is a copy of another run excluded (config.BROKEN_EVENT_RUNS):
  - Fig. 6: parse -> cluster GLM -> visualize   (new16 IDs 7/10)
      7 = television news interviews (6a), 10 = natural landscapes (6b)
The arXiv v1/v2 maps (new54 IDs 5/32/49, 6/30/31) are reproduced with
analysis/cluster_contrasts.py --family new54 directly (README).

The engine is analysis/glm_engine.py + regressors.py (pure numpy/scipy/nibabel, no nilearn).
Reproduces the PUBLISHED n=78 / df=77 analysis (5 canonical subjects drop in the confound
loader — docs/DESIGN.md §7). Precomputed cut = fsaverage6 GIFTIs + confounds (--derivatives-root).

--input-source precomputed (default): the Tier-1 cut is the shipped per-subject cluster
  t-maps (cluster_contrasts_new16clusters_exclbrokenruns/subject_level/, ~62 MB), NOT the ~1.5 TB
  fsaverage6 BOLD. `--from-subject-maps` loads those and runs only the deterministic group
  ttest+FDR → render (docs/DESIGN.md §5.1-C). Light — runs on the login node. Uses the vendored
  cluster CSVs (data/cluster_assignments/); no raw download, no fMRIPrep, no per-subject GLM.
--input-source raw: (re)derives the cluster CSVs from the raw BIDS events via
  analysis/parse_clusters.py, then re-fits the per-subject GLM from the fsaverage6 BOLD
  (regenerating subject_level/) before the group stage. The per-subject GLM loads
  78×13-run fsaverage6 BOLD — memory/IO-heavy; run on a compute node, not the login node
  (README §3). The raw fMRIPrep preprocessing itself (Stage 0) is a separate,
  container-based step (preprocessing/), faithful-ported and NOT golden-mastered (§2.5/§6).
The visualize step requires nilearn (pinned 0.12.1); the GLM/group step does not.

STATUS: the Stage-1 figure lineage (parse -> cluster GLM -> visualize) is wired and ported,
visualizer included. Stage-0 raw preprocessing is a faithful container port (not
bitwise-reproducible, so not golden-mastered — docs/DESIGN.md §2.5/§6). The one open item is
heavy group-map golden tolerance calibration (tests/calibrate_heavy_golden.py, SLURM).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import config

FIGURES = tuple(config.FIGURES)

_HERE = Path(__file__).resolve().parent
_ANALYSIS = _HERE / "analysis"

# figure key -> (cluster family, vendored model JSON, cluster IDs, exclude broken runs).
# config.FIGURES is the single source of truth.
_FIG_SPEC = {
    key: (fig["family"],
          config.CLUSTERS[fig["family"]]["model_json"].rsplit("/", 1)[-1],
          fig["ids"],
          fig["exclude_broken_runs"])
    for key, fig in config.FIGURES.items()
}

# Paper-figure tag folded into rendered filenames (config.FIGURES has the labels).
_FIG_TAG = {"fig6": "Fig6"}


def _load(module_filename: str):
    path = _ANALYSIS / module_filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-source", choices=("precomputed", "raw"), default="precomputed")
    p.add_argument("--derivatives-root", default=None, help="fsaverage6 GIFTIs + confounds (the precomputed cut).")
    p.add_argument("--results-root", default=None, help="Where group GLMs / intermediate maps are written (default: --derivatives-root).")
    p.add_argument("--plots-root", default=str(_HERE / "plots"),
                   help="Directory for rendered figures (default: <dataset>/plots). Group-level GLM "
                        "maps still go under --results-root.")
    p.add_argument("--raw-root", default=None, help="OpenNeuro ds005256 BIDS root (events; required for --input-source raw and for the GLM's run-relative onsets).")
    p.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    p.add_argument("--skip-visualize", action="store_true", help="Run the GLM branches only; skip HTML rendering (which needs nilearn).")
    return p


def _resolve_csv(figure_key: str, input_source: str, raw_root, results_root) -> str:
    """Return the cluster-assignments CSV path, (re)generating it on the raw path."""
    family, json_name, _, _ = _FIG_SPEC[figure_key]
    vendored_csv = _HERE / config.CLUSTERS[family]["assignments_csv"]

    if input_source == "precomputed":
        return str(vendored_csv)

    # raw: regenerate the CSV from the vendored model JSON + raw sub-0001 events.
    if not raw_root:
        raise SystemExit("--input-source raw requires --raw-root (for sub-0001 events).")
    out_csv = Path(results_root) / "cluster_assignments" / vendored_csv.name
    _load("parse_clusters.py").main([
        "--json-file", str(_HERE / config.CLUSTER_ASSIGNMENTS_DIR / json_name),
        "--raw-root", str(raw_root),
        "--out-csv", str(out_csv),
    ])
    return str(out_csv)


def _run_figure(figure_key: str, args) -> None:
    family, _, cluster_ids, exclude_broken_runs = _FIG_SPEC[figure_key]
    derivatives_root = args.derivatives_root
    if not derivatives_root:
        raise SystemExit("--derivatives-root is required (the precomputed fsaverage6 cut).")
    # The GLM needs run-relative video onsets from events.tsv; default to derivatives_root's
    # sibling raw tree if --raw-root not given (events live in the raw BIDS dataset).
    bids_root = args.raw_root or str(Path(derivatives_root).parent)
    results_root = args.results_root or derivatives_root

    csv = _resolve_csv(figure_key, args.input_source, args.raw_root, results_root)

    contrasts = _load("cluster_contrasts.py")
    subdir = contrasts.FAMILIES[family]["derivatives_subdir"]
    if exclude_broken_runs:
        subdir += config.EXCLUDED_RUNS_SUFFIX
    output_dir = Path(results_root) / subdir

    for cid in cluster_ids:
        contrast_argv = [
            "--family", family,
            "--cluster-id", str(cid),
            "--cluster-file", csv,
            "--derivatives-root", derivatives_root,
            "--bids-root", bids_root,
            "--output-dir", str(output_dir),
        ]
        if exclude_broken_runs:
            contrast_argv.append("--exclude-broken-runs")
        # Precomputed cut = shipped subject-level t-maps (the Tier-1 cut, ~62 MB for the 2
        # Fig. 6 clusters), NOT the ~1.5 TB fsaverage6 BOLD. Render from those instead of
        # re-fitting the GLM (docs/DESIGN.md §5.1-C). The cut's subject_level/ lives under
        # derivatives-root/<subdir> (read-only); group maps + plots write to output_dir
        # (results-root/<subdir>). --input-source raw regenerates subject_level/ from BOLD.
        if args.input_source == "precomputed":
            contrast_argv += ["--from-subject-maps",
                              "--subject-maps-root", str(Path(derivatives_root) / subdir)]
        contrasts.main(contrast_argv)

    if not args.skip_visualize:
        viz = _load("visualize_clusters.py")
        labels = config.FIGURES[figure_key]["labels"]
        viz.main([
            "--contrast-dir", str(output_dir / "group_level"),
            "--output-dir", str(Path(args.plots_root) / subdir),
            "--cluster-ids", *[str(c) for c in cluster_ids],
            "--figure-tag", _FIG_TAG[figure_key],
            "--cluster-labels", *[f"{c}={labels[c]}" for c in cluster_ids],
        ])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for figure_key in args.figures:
        _run_figure(figure_key, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
