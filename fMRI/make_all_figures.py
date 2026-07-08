#!/usr/bin/env python
"""Meta script — regenerate every brain-side paper figure in one run.

Thin orchestrator (docs/DESIGN.md §3): dispatches each dataset's ``make_figures.py`` in
``--input-source precomputed`` mode by default, so a single command reproduces all three
datasets' figures for AlKhamissi & Mehrer et al., 2026 (Pernet 2015, Marvi 2025, Jung 2025).

    python make_all_figures.py --derivatives-root <CUT_BASE>

Why subprocess dispatch (not in-process import)?  Two hard constraints (docs/DESIGN.md §6):
  1. **Module-name collision.** Each dataset's ``make_figures.py`` — and the analysis
     modules it loads — does a bare ``import config`` resolving to *its own* ``config.py``.
     Importing two datasets into one interpreter would clobber ``config`` (and ``analysis``/
     ``preprocessing``/``emfl``). So each runs in its own process with cwd = the dataset dir.
  2. **Divergent Stage-1 envs.** Pernet is pinned to nilearn 0.10.4; Marvi/Jung to 0.12.1
     (different numpy again between them). No single interpreter satisfies all three, so the
     interpreter is selectable *per dataset* (``--python NAME=PATH``), defaulting to the one
     running this script. Conda envs: ``pernet_2015_env`` / ``omni-fmri-marvi`` / ``omni-fmri-jung``.

**Flag divergence (docs/DESIGN.md §1).**  Marvi & Jung resolve their precomputed cut via
``--derivatives-root``; **Pernet has no --derivatives-root — its cut is contrast-level, passed
via --results-root.**  This orchestrator hides that: it knows each dataset's cut-flag and maps
a single ``--derivatives-root <BASE>`` (the downloaded/hosted tier) to ``<BASE>/<Dataset>``,
handing it to the right flag.  Point a dataset at an arbitrary on-disk path with
``--root NAME=PATH`` (used for the on-disk sanity check — the three dev cuts live at unrelated
absolute paths, not under one base).

Each dataset's ``make_figures.py`` is also runnable standalone (see its ``--help``).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _subprocess_env():
    """Env for the dataset subprocesses with the repo root on PYTHONPATH so the shared top-level
    ``core`` package (e.g. ``core.surface`` used by Pernet's 02_surface_projection) imports when
    cwd is the dataset dir. Runtime equivalent of conftest.py's sys.path insert — needed until
    ``core`` is pip-installed per env. The dataset dir stays sys.path[0] (Python prepends the
    script dir), so each dataset's own ``config``/``preprocessing`` still win."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(_HERE), env["PYTHONPATH"]]) if env.get("PYTHONPATH") else str(_HERE)
    return env

# Per-dataset dispatch registry. Order follows the build/port order in docs/DESIGN.md §9.
#   root_flag     : the flag THIS dataset's make_figures.py uses for its precomputed cut.
#   results_is_cut: True when the cut dir and the figure-output dir are the same directory
#                   (Pernet: figures land alongside the contrast maps under --results-root,
#                   so there is no separate --results-root to redirect). Marvi/Jung write
#                   figures to a --results-root that defaults to --derivatives-root.
#   conda_env     : the intended conda env name (documentation for --python selection).
DATASETS = {
    "Pernet_2015": {"root_flag": "--results-root", "results_is_cut": True, "conda_env": "pernet_2015_env"},
    "Marvi_2025": {"root_flag": "--derivatives-root", "results_is_cut": False, "conda_env": "omni-fmri-marvi"},
    "Jung_2025": {"root_flag": "--derivatives-root", "results_is_cut": False, "conda_env": "omni-fmri-jung"},
}


def _parse_mapping(values, flag_name):
    """Parse repeated ``NAME=VALUE`` overrides into a {dataset: value} dict, validating names."""
    out = {}
    for item in values or ():
        if "=" not in item:
            raise SystemExit(f"{flag_name} expects NAME=VALUE (got {item!r}).")
        name, _, value = item.partition("=")
        if name not in DATASETS:
            raise SystemExit(f"{flag_name}: unknown dataset {name!r}. Choose from {list(DATASETS)}.")
        out[name] = value
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-source", choices=("precomputed", "raw"), default="precomputed",
                   help="Start from the hosted precomputed cut (default) or run the full raw pipeline.")
    p.add_argument("--derivatives-root", default=None,
                   help="Base dir holding each dataset's precomputed cut as <BASE>/<Dataset>. "
                        "Mapped to each dataset's own cut flag (Pernet: --results-root; others: --derivatives-root).")
    p.add_argument("--results-root", default=None,
                   help="Base dir for figure outputs as <BASE>/<Dataset> (Marvi/Jung only; Pernet writes "
                        "into its --results-root cut). Defaults to the cut location.")
    p.add_argument("--raw-root", default=None,
                   help="Base dir for raw BIDS as <BASE>/<Dataset> (only for --input-source raw).")
    p.add_argument("--root", action="append", metavar="NAME=PATH", default=[],
                   help="Override the resolved cut path for one dataset (repeatable). Use for arbitrary "
                        "on-disk paths that don't sit under --derivatives-root.")
    p.add_argument("--results", action="append", metavar="NAME=PATH", default=[],
                   help="Override the figure-output path for one dataset (repeatable; Marvi/Jung).")
    p.add_argument("--raw", action="append", metavar="NAME=PATH", default=[],
                   help="Override the raw-BIDS path for one dataset (repeatable).")
    p.add_argument("--python", action="append", metavar="NAME=PATH", default=[],
                   help="Per-dataset Python interpreter (repeatable) — needed because the three Stage-1 envs "
                        "diverge (nilearn 0.10.4 vs 0.12.1). Defaults to this interpreter for every dataset.")
    p.add_argument("--figures", action="append", metavar="NAME=fig1,fig2", default=[],
                   help="Restrict one dataset to a subset of its figures (repeatable). Default: all figures.")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS),
                   help="Subset of datasets to run (default: all three).")
    p.add_argument("--download-first", action="store_true",
                   help="Fetch each dataset's precomputed tier from OSF before running (docs/DESIGN.md §5). "
                        "Not wired in this orchestrator — run download_precomputed.py first, then point "
                        "--derivatives-root at its output.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the per-dataset command that would run (with cwd + interpreter) and exit — "
                        "the dispatch sanity check (docs/DESIGN.md §9) without needing every env live.")
    return p


def _resolve_root(name, override_map, base, subdir_of_base=True):
    """Resolve a per-dataset path: explicit override wins, else <base>/<name>, else None."""
    if name in override_map:
        return override_map[name]
    if base is None:
        return None
    return str(Path(base) / name) if subdir_of_base else base


def build_command(name, args, roots, results, raws, pythons, figure_sets):
    """Build (interpreter, argv, cwd) for one dataset, or raise SystemExit on missing inputs."""
    spec = DATASETS[name]
    cut = _resolve_root(name, roots, args.derivatives_root)
    if cut is None:
        raise SystemExit(
            f"{name}: no precomputed cut location. Pass --derivatives-root <BASE> (uses <BASE>/{name}) "
            f"or --root {name}=<PATH>.")

    argv = ["make_figures.py", "--input-source", args.input_source, spec["root_flag"], cut]

    # Figure-output redirection (Marvi/Jung expose a separate --results-root; Pernet does not).
    if not spec["results_is_cut"]:
        out = _resolve_root(name, results, args.results_root)
        if out is not None:
            argv += ["--results-root", out]

    # Raw BIDS root (only meaningful for --input-source raw; harmless otherwise but we omit it).
    if args.input_source == "raw":
        raw = _resolve_root(name, raws, args.raw_root)
        if raw is None:
            raise SystemExit(f"{name}: --input-source raw needs a raw root (--raw-root <BASE> or --raw {name}=<PATH>).")
        argv += ["--raw-root", raw]

    if name in figure_sets:
        argv += ["--figures", *figure_sets[name]]

    interpreter = pythons.get(name, sys.executable)
    cwd = _HERE / name
    return interpreter, argv, cwd


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.download_first:
        raise NotImplementedError(
            "--download-first is not wired into this orchestrator. Run download_precomputed.py first "
            "(see docs/OSF_DATA.md), then pass --derivatives-root <dest>.")

    roots = _parse_mapping(args.root, "--root")
    results = _parse_mapping(args.results, "--results")
    raws = _parse_mapping(args.raw, "--raw")
    pythons = _parse_mapping(args.python, "--python")
    figure_sets = {n: v.split(",") for n, v in _parse_mapping(args.figures, "--figures").items()}

    plans = [(name, *build_command(name, args, roots, results, raws, pythons, figure_sets))
             for name in args.datasets]

    failures = []
    for name, interpreter, script_argv, cwd in plans:
        cmd = [interpreter, *script_argv]
        print(f"\n=== {name} ({DATASETS[name]['conda_env']}) ===")
        print(f"    cwd: {cwd}")
        print(f"    cmd: {' '.join(cmd)}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=str(cwd), env=_subprocess_env())
        if result.returncode != 0:
            failures.append((name, result.returncode))
            print(f"    ✗ {name} exited {result.returncode}")

    if args.dry_run:
        print("\n(dry run — nothing executed)")
        return 0
    if failures:
        print("\nFAILED datasets: " + ", ".join(f"{n} (exit {rc})" for n, rc in failures))
        return 1
    print("\nAll datasets reproduced their figures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
