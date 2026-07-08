#!/usr/bin/env python3
"""Driver: run the cluster-vs-others GLM for one cluster of the new54 family.

Path-agnostic port of dev `src/45_new54cluster_contrasts_standard.py` — a thin wrapper
over the shared engine (20251211_fMRI_movie_watching_spacetop @ 4066746). All six
published maps are single discovered clusters of the one new54 (54-cluster) partition
(App. D "14 clusters" is a typo for 54 — README §4); the dev 14-/21-/22-cluster +
supercluster wrappers are NOT ported (docs/DESIGN.md §8).

  - ``--family new54`` → cluster_contrasts_new54clusters/
      IDs 5/32/49 → Fig. 6 / D4  ·  IDs 6/30/31 → Fig. D5

Reads the fsaverage6 BOLD + confounds under ``--derivatives-root`` and the (raw)
``task-alignvideo`` events under ``--bids-root`` (needed for run-relative video onsets);
writes per-subject and group GIFTIs under ``--output-dir``. Reproduces the published
n=78 / df=77 maps (5 canonical subjects drop in the confound loader — docs/DESIGN.md §7).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm_engine import run_full_analysis  # noqa: E402

FAMILIES = {
    'new54': {'derivatives_subdir': 'cluster_contrasts_new54clusters'},
}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--family', choices=tuple(FAMILIES), default='new54',
                   help='Cluster family (only new54 is published).')
    p.add_argument('--cluster-id', type=int, required=True, help='Target cluster ID.')
    p.add_argument('--cluster-file', required=True, help='Cluster-assignments CSV (from parse_clusters).')
    p.add_argument('--derivatives-root', required=True,
                   help='fsaverage6 GIFTIs + confounds (the precomputed cut).')
    p.add_argument('--bids-root', required=True,
                   help='BIDS root with task-alignvideo events.tsv (run-relative video onsets).')
    p.add_argument('--output-dir', default=None,
                   help='Output dir (default: <derivatives-root>/<family subdir>).')
    p.add_argument('--subjects', nargs='*', default=None,
                   help='Subject IDs to process (default: all 83 canonical).')
    p.add_argument('--from-subject-maps', action='store_true',
                   help='Reuse shipped subject-level t-maps instead of re-fitting the '
                        'per-subject GLM from fsaverage6 BOLD — the precomputed-cut / '
                        'render-only path (docs/DESIGN.md §5.1-C). Only the group ttest+FDR runs; '
                        '--derivatives-root/--bids-root BOLD are not read.')
    p.add_argument('--subject-maps-root', default=None,
                   help='Dir CONTAINING subject_level/ for --from-subject-maps (the read-only '
                        'cut). Default: --output-dir. Set this when outputs go to a separate '
                        '--output-dir / --results-root from the shipped subject-level cut.')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    cluster_file = args.cluster_file
    df = pd.read_csv(cluster_file)
    id_min, id_max = int(df['cluster_id'].min()), int(df['cluster_id'].max())
    if not id_min <= args.cluster_id <= id_max:
        print(f"ERROR: --cluster-id must be {id_min}-{id_max} for family '{args.family}', got {args.cluster_id}")
        return 1

    cluster_label = df[df['cluster_id'] == args.cluster_id].iloc[0]['cluster_label']

    output_dir = args.output_dir or (
        Path(args.derivatives_root) / FAMILIES[args.family]['derivatives_subdir']
    )
    print(f"Family:  {args.family}")
    print(f"Cluster: {args.cluster_id} ({cluster_label})")
    print(f"Output:  {output_dir}")

    success = run_full_analysis(
        target_id=args.cluster_id,
        output_dir=output_dir,
        bold_dir=args.derivatives_root,
        confounds_dir=args.derivatives_root,
        bids_dir=args.bids_root,
        cluster_file=cluster_file,
        subjects=args.subjects,
        reuse_subject_maps=args.from_subject_maps,
        subject_maps_root=args.subject_maps_root,
    )

    if success:
        print(f"\n✓ Completed: cluster {args.cluster_id} ({cluster_label})")
        return 0
    print(f"\n✗ Failed: cluster {args.cluster_id} ({cluster_label})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
