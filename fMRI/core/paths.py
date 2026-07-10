"""Path & input-source resolution — the shared CLI convention (docs/DESIGN.md §2.3, §4).

Genuinely NEW code (not a port) -> real TDD applies here (docs/DESIGN.md §2.6, §6 Tier 2).
Resolves the `--input-source {precomputed, raw}` switch plus `--raw-root`,
`--derivatives-root`, `--results-root` into concrete paths, uniformly across all
three datasets. Pernet has no `--derivatives-root` (its precomputed cut is
contrast-level -> `--results-root`); this module encodes that per-dataset variation.

STATUS: scaffold — write the TDD tests in core/tests first, then implement.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedPaths:
    """Resolved locations for one dataset run."""
    input_source: str            # "precomputed" | "raw"
    raw_root: str | None
    derivatives_root: str | None  # None for Pernet (contrast-level cut)
    results_root: str | None


def add_common_arguments(parser: argparse.ArgumentParser, *, has_derivatives: bool = True) -> argparse.ArgumentParser:
    """Attach the shared `--input-source` / `--*-root` flags to a dataset parser.

    TODO(new): implement + TDD. `has_derivatives=False` for Pernet.
    """
    raise NotImplementedError("core.paths.add_common_arguments — scaffold (docs/DESIGN.md §2.3)")


def resolve(args: argparse.Namespace, *, has_derivatives: bool = True) -> ResolvedPaths:
    """Turn parsed args into a validated `ResolvedPaths`.

    TODO(new): implement + TDD (resolution rules are the unit-test target, §6 Tier 2).
    """
    raise NotImplementedError("core.paths.resolve — scaffold (docs/DESIGN.md §2.3)")
