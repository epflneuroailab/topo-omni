#!/usr/bin/env python3
"""Tabulate a cluster family's group-level results: one row per cluster.

Reads the family's cluster-assignment CSV and its ``group_level/`` maps (written by
``cluster_contrasts.py``) and reports, per cluster:

  * content: label, #clips, #videos, the videos, and ``n_unmodeled_clips`` — clips of the
    target cluster's videos that were assigned to OTHER clusters. The published design
    (``regressors.create_cluster_regressors``) marks as "others" only videos NOT in the
    target cluster, so those clips are in neither regressor (implicit baseline).
  * group stats: max t, the one-tailed BH-FDR q<.05 threshold computed jointly over both
    hemispheres (``visualize_clusters.calculate_fdr_threshold`` — the paper's stated
    procedure and exactly what the figures threshold on), and the survivor count
    ``#(t >= threshold)``.
  * n subjects entering the group test, from the per-cluster summary JSON when present
    (else assumed = expected, ``n_subjects_source='assumed'``); a mismatch is flagged in
    ``status`` because the engine drops a subject on ANY error.
  * ``excluded_runs``: the copied-events runs dropped for included subjects (from the
    per-cluster summary; empty for the published analysis).
  * optionally, the best-matching reference cluster (e.g. published new54) by clip Jaccard.

Usage (new27 rebuttal family, broken runs excluded, vs the published new54 clusters):

    python analysis/summarize_family.py --family new27 --exclude-broken-runs \\
        --results-root <DERIV> --reference-family new54 --reference-ids 5 32 49 6 30 31

Writes ``<results-root>/<family subdir>/summary_table.{csv,md}`` unless --out-prefix is given.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_clusters import calculate_fdr_threshold  # noqa: E402

_DATASET = Path(__file__).resolve().parent.parent
FDR_Q = 0.05


def _content_table(assign: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, grp in assign.groupby("cluster_id", sort=True):
        videos = sorted(grp["video_name"].unique())
        others = assign[assign["cluster_id"] != cid]
        rows.append({
            "cluster_id": int(cid),
            "label": grp["cluster_label"].iloc[0],
            "n_clips": len(grp),
            "n_videos": len(videos),
            "videos": ";".join(videos),
            "n_unmodeled_clips": int(others["video_name"].isin(videos).sum()),
        })
    return pd.DataFrame(rows)


def _load_tstat(group_dir: Path, cid: int, hemi: str):
    path = group_dir / f"group_cluster-{cid:02d}_space-fsaverage6_hemi-{hemi}_tstat.func.gii"
    return nib.load(str(path)).darrays[0].data if path.exists() else None


def _group_stats(group_dir: Path, cid: int, n_expected: int) -> dict:
    lh, rh = _load_tstat(group_dir, cid, "L"), _load_tstat(group_dir, cid, "R")
    if lh is None or rh is None:
        return {"status": "missing"}

    summary_path = group_dir / f"group_cluster-{cid:02d}_space-fsaverage6_summary.json"
    excluded = ""
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        n_subj, source = int(summary["n_subjects"]), "summary"
        excluded = ";".join(f"{s}:{ses}/{run}" for s, ses, run in summary.get("excluded_runs", []))
    else:
        n_subj, source = n_expected, "assumed"

    fdr_t, _, _ = calculate_fdr_threshold(lh, rh, q=FDR_Q, df=n_subj - 1)
    both = np.concatenate([lh, rh])
    return {
        "status": "done" if n_subj == n_expected else f"n_subjects!={n_expected}",
        "n_subjects": n_subj,
        "n_subjects_source": source,
        "max_t": float(both.max()),
        "fdr_t": float(fdr_t),
        "n_fdr_survivors": int((both >= fdr_t).sum()) if np.isfinite(fdr_t) else 0,
        "excluded_runs": excluded,
    }


def _best_reference(assign: pd.DataFrame, ref: pd.DataFrame, ref_ids) -> pd.DataFrame:
    ref_sets = {int(r): set(ref.loc[ref["cluster_id"] == r, "clip_filename"]) for r in ref_ids}
    rows = []
    for cid, grp in assign.groupby("cluster_id", sort=True):
        clips = set(grp["clip_filename"])
        scores = {r: len(clips & s) / len(clips | s) for r, s in ref_sets.items()}
        best = max(scores, key=scores.get)
        rows.append({"cluster_id": int(cid), "best_ref_id": best, "best_ref_jaccard": scores[best]})
    return pd.DataFrame(rows)


def summarize(assignments_csv, group_dir, n_subjects_expected=78,
              reference_csv=None, reference_ids=None) -> pd.DataFrame:
    """Return the per-cluster table (see module docstring)."""
    assign = pd.read_csv(assignments_csv)
    table = _content_table(assign)
    stats = pd.DataFrame([{"cluster_id": cid, **_group_stats(Path(group_dir), cid, n_subjects_expected)}
                          for cid in table["cluster_id"]])
    table = table.merge(stats, on="cluster_id")
    if reference_csv is not None:
        ref = pd.read_csv(reference_csv)
        ids = reference_ids if reference_ids else sorted(ref["cluster_id"].unique())
        table = table.merge(_best_reference(assign, ref, ids), on="cluster_id")
    return table


def to_markdown(table: pd.DataFrame) -> str:
    """Plain GitHub-markdown table (no `tabulate` dependency)."""
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.2f}"
        return str(v)
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _config():
    spec = importlib.util.spec_from_file_location("jung_config_summarize", _DATASET / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, help="Cluster family key in config.CLUSTERS (e.g. new27).")
    p.add_argument("--results-root", required=True,
                   help="Dir containing <family subdir>/group_level/ (the cluster_contrasts --output-dir parent).")
    p.add_argument("--n-subjects", type=int, default=78, help="Expected n in the group test (published: 78).")
    p.add_argument("--exclude-broken-runs", action="store_true",
                   help="Read the <family subdir>_exclbrokenruns results written by "
                        "cluster_contrasts --exclude-broken-runs (the SLURM script's default).")
    p.add_argument("--reference-family", default=None, help="Family to match clusters against (e.g. new54).")
    p.add_argument("--reference-ids", nargs="*", type=int, default=None,
                   help="Reference cluster IDs to consider (default: all).")
    p.add_argument("--out-prefix", default=None,
                   help="Output path prefix (default: <results-root>/<family subdir>/summary_table).")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = _config()
    fam = cfg.CLUSTERS[args.family]
    suffix = cfg.EXCLUDED_RUNS_SUFFIX if args.exclude_broken_runs else ""
    family_dir = Path(args.results_root) / (fam["derivatives_subdir"] + suffix)
    ref_csv = (_DATASET / cfg.CLUSTERS[args.reference_family]["assignments_csv"]
               if args.reference_family else None)

    table = summarize(_DATASET / fam["assignments_csv"], family_dir / "group_level",
                      n_subjects_expected=args.n_subjects,
                      reference_csv=ref_csv, reference_ids=args.reference_ids)

    prefix = Path(args.out_prefix) if args.out_prefix else family_dir / "summary_table"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(f"{prefix}.csv", index=False)
    Path(f"{prefix}.md").write_text(to_markdown(table))
    print(to_markdown(table))
    print(f"Wrote {prefix}.csv and {prefix}.md")
    n_bad = int((table["status"] != "done").sum())
    if n_bad:
        print(f"WARNING: {n_bad} cluster(s) not 'done' — see the status column.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
