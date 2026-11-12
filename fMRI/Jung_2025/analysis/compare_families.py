#!/usr/bin/env python3
"""Compare a cluster family's group maps with a reference family's (clips and maps).

For every pair (cluster ``a`` of ``--family``, cluster ``b`` of ``--reference-family``) that
shares at least one clip, reports:

  * clip overlap: ``n_shared``, Jaccard, ``frac_a_in_b`` = |a∩b|/|a|, ``frac_b_in_a``, and
    ``relation`` (``identical`` / ``a_contains_b`` / ``b_contains_a`` / ``partial``);
  * map similarity, when both group maps exist: Pearson ``map_r`` of the whole-brain
    unthresholded t-maps (both hemispheres), and the Dice of the FDR-survivor masks
    (``dice_fdr``) and of the top-10%-of-FDR-survivor masks (``dice_top10``) — the one-tailed
    BH-FDR q<.05 threshold computed jointly over both hemispheres with df = n-1, i.e. exactly
    what the figures and ``summarize_family.py`` threshold on. ``dice_top10_all`` uses the top
    10% of all vertices instead (8,192 of 81,924), so it is defined for null maps too;
  * both clusters' ``max_t`` and FDR-survivor counts.

A second, per-cluster table gives, for each ``a``, the best reference cluster by clips and the
best by map correlation over ALL reference maps found (shared clips or not), plus, for
clusters that merge several reference clusters, the mean map r to those parts.

``--all-pairs`` reports every pair, including those sharing no clips (``relation`` =
``disjoint``); with ``--ids`` it gives a small targeted table, e.g. one new16 cluster against the
six paper maps.

``identical`` pairs have the same target clips, hence the same GLM design, so their maps must
agree up to float noise (``map_r`` ~ 1): a free check that the run matched the earlier one.

Usage (new16 vs the new27 family it coarsens, and vs the six published new54 maps):

    python analysis/compare_families.py --family new16 --reference-family new27 \\
        --exclude-broken-runs --results-root <DERIV>
    python analysis/compare_families.py --family new16 --reference-family new54 \\
        --exclude-broken-runs --results-root <DERIV> --reference-ids 5 32 49 6 30 31
    # one cluster against chosen reference maps, whether or not they share clips
    python analysis/compare_families.py --family new16 --ids 7 --reference-family new54 \
        --exclude-broken-runs --results-root <DERIV> --reference-ids 5 32 49 6 30 31 \
        --all-pairs --no-plot --out-prefix <DERIV>/.../compare_c07_vs_new54paper

Writes ``<results-root>/<family subdir>/compare_<reference>.{csv,md}``,
``compare_<reference>_per_cluster.{csv,md}`` and ``compare_<reference>_map_r.png`` unless
--out-prefix is given.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_family import _config, _load_tstat, to_markdown  # noqa: E402
from visualize_clusters import calculate_fdr_threshold, top_of_fdr_threshold  # noqa: E402

_DATASET = Path(__file__).resolve().parent.parent
FDR_Q = 0.05
TOP_PERCENTILE = 10.0


def load_map(group_dir: Path, cid: int, n_default: int = 78):
    """Group t-map (L+R concatenated) + FDR / top-10% masks, or None if not on disk."""
    lh, rh = _load_tstat(group_dir, cid, "L"), _load_tstat(group_dir, cid, "R")
    if lh is None or rh is None:
        return None
    summary = group_dir / f"group_cluster-{cid:02d}_space-fsaverage6_summary.json"
    n = int(json.loads(summary.read_text())["n_subjects"]) if summary.exists() else n_default
    fdr_t, _, _ = calculate_fdr_threshold(lh, rh, q=FDR_Q, df=n - 1)
    top_t, _, _ = top_of_fdr_threshold(lh, rh, TOP_PERCENTILE, df=n - 1, q=FDR_Q)
    t = np.concatenate([lh, rh]).astype(np.float64)
    top_all_t = np.nanpercentile(t, 100.0 - TOP_PERCENTILE)
    return {
        "t": t, "n": n, "max_t": float(t.max()),
        "fdr": t >= fdr_t if np.isfinite(fdr_t) else np.zeros(t.shape, bool),
        "top": t >= top_t if np.isfinite(top_t) else np.zeros(t.shape, bool),
        "top_all": t >= top_all_t,
    }


def dice(m1: np.ndarray, m2: np.ndarray) -> float:
    denom = int(m1.sum()) + int(m2.sum())
    return float("nan") if denom == 0 else 2.0 * int((m1 & m2).sum()) / denom


def map_r(t1: np.ndarray, t2: np.ndarray) -> float:
    ok = np.isfinite(t1) & np.isfinite(t2)
    return float(np.corrcoef(t1[ok], t2[ok])[0, 1])


def _relation(n_a, n_b, n_shared):
    if n_shared == 0:
        return "disjoint"
    if n_shared == n_a == n_b:
        return "identical"
    if n_shared == n_b:
        return "a_contains_b"
    if n_shared == n_a:
        return "b_contains_a"
    return "partial"


def compare(assign_csv, group_dir, ref_csv, ref_group_dir, ref_ids=None, ids=None,
            all_pairs=False):
    """Return (pair table, per-cluster table, full A x B map-r matrix).

    ``ids`` / ``ref_ids`` restrict either side; ``all_pairs`` keeps pairs sharing no clips.
    """
    a_df, b_df = pd.read_csv(assign_csv), pd.read_csv(ref_csv)
    a_sets = {int(c): set(g["clip_filename"]) for c, g in a_df.groupby("cluster_id")}
    b_sets = {int(c): set(g["clip_filename"]) for c, g in b_df.groupby("cluster_id")}
    a_lab = a_df.groupby("cluster_id")["cluster_label"].first().to_dict()
    b_lab = b_df.groupby("cluster_id")["cluster_label"].first().to_dict()
    if ref_ids:
        b_sets = {b: s for b, s in b_sets.items() if b in set(ref_ids)}
    if ids:
        a_sets = {a: s for a, s in a_sets.items() if a in set(ids)}

    a_maps = {a: load_map(Path(group_dir), a) for a in a_sets}
    b_maps = {b: load_map(Path(ref_group_dir), b) for b in b_sets}
    r_mat = pd.DataFrame(
        [[map_r(a_maps[a]["t"], b_maps[b]["t"]) if a_maps[a] and b_maps[b] else np.nan
          for b in b_sets] for a in a_sets],
        index=pd.Index(list(a_sets), name="cluster_id"), columns=list(b_sets))

    pairs = []
    for a, sa in a_sets.items():
        for b, sb in b_sets.items():
            shared = len(sa & sb)
            if not shared and not all_pairs:
                continue
            ma, mb = a_maps[a], b_maps[b]
            both = ma is not None and mb is not None
            pairs.append({
                "cluster_id": a, "label": a_lab[a], "ref_id": b, "ref_label": b_lab[b],
                "n_clips": len(sa), "ref_n_clips": len(sb), "n_shared": shared,
                "jaccard": shared / len(sa | sb),
                "frac_a_in_b": shared / len(sa), "frac_b_in_a": shared / len(sb),
                "relation": _relation(len(sa), len(sb), shared),
                "map_r": r_mat.loc[a, b],
                "dice_fdr": dice(ma["fdr"], mb["fdr"]) if both else np.nan,
                "dice_top10": dice(ma["top"], mb["top"]) if both else np.nan,
                "dice_top10_all": dice(ma["top_all"], mb["top_all"]) if both else np.nan,
                "max_t": ma["max_t"] if ma else np.nan,
                "ref_max_t": mb["max_t"] if mb else np.nan,
                "n_fdr": int(ma["fdr"].sum()) if ma else -1,
                "ref_n_fdr": int(mb["fdr"].sum()) if mb else -1,
            })
    pairs = pd.DataFrame(pairs)

    per = []
    for a in a_sets:
        mine = pairs[(pairs["cluster_id"] == a) & (pairs["n_shared"] > 0)] if len(pairs) else pairs
        best_clip = mine.loc[mine["jaccard"].idxmax()] if len(mine) else None
        row_r = r_mat.loc[a].dropna()
        parts = mine[mine["relation"] == "a_contains_b"]
        per.append({
            "cluster_id": a, "label": a_lab[a],
            "n_fdr": int(a_maps[a]["fdr"].sum()) if a_maps[a] else -1,
            "max_t": a_maps[a]["max_t"] if a_maps[a] else np.nan,
            "best_clip_ref_id": int(best_clip["ref_id"]) if best_clip is not None else -1,
            "best_clip_jaccard": float(best_clip["jaccard"]) if best_clip is not None else np.nan,
            "best_clip_relation": best_clip["relation"] if best_clip is not None else "",
            "best_clip_map_r": float(best_clip["map_r"]) if best_clip is not None else np.nan,
            "best_map_ref_id": int(row_r.idxmax()) if len(row_r) else -1,
            "best_map_r": float(row_r.max()) if len(row_r) else np.nan,
            "n_parts": len(parts),
            "parts": ";".join(str(int(b)) for b in parts["ref_id"]),
            "mean_map_r_to_parts": float(parts["map_r"].mean()) if len(parts) else np.nan,
        })
    return pairs, pd.DataFrame(per), r_mat


def plot_map_r(r_mat: pd.DataFrame, out_png, family, reference):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(0.32 * r_mat.shape[1] + 2.5, 0.32 * r_mat.shape[0] + 1.8))
    im = ax.imshow(r_mat.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(r_mat.shape[1]), [str(c) for c in r_mat.columns], fontsize=7, rotation=90)
    ax.set_yticks(range(r_mat.shape[0]), [str(c) for c in r_mat.index], fontsize=7)
    ax.set_xlabel(f"{reference} cluster")
    ax.set_ylabel(f"{family} cluster")
    ax.set_title(f"Group t-map correlation, {family} vs {reference}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, help="Family to describe (e.g. new16).")
    p.add_argument("--reference-family", required=True, help="Family to compare against (e.g. new27, new54).")
    p.add_argument("--results-root", required=True,
                   help="Dir containing both families' <subdir>/group_level/.")
    p.add_argument("--exclude-broken-runs", action="store_true",
                   help="Read the *_exclbrokenruns results of both families.")
    p.add_argument("--reference-group-dir", default=None,
                   help="Override the reference group_level/ dir (e.g. the published new54 "
                        "maps, which exist for all 54 clusters without the exclusion).")
    p.add_argument("--reference-ids", nargs="*", type=int, default=None,
                   help="Reference cluster IDs to consider (default: all).")
    p.add_argument("--ids", nargs="*", type=int, default=None,
                   help="Cluster IDs of --family to describe (default: all).")
    p.add_argument("--all-pairs", action="store_true",
                   help="Report every pair, also those sharing no clips (relation 'disjoint').")
    p.add_argument("--out-prefix", default=None,
                   help="Output prefix (default: <results-root>/<family subdir>/compare_<reference>).")
    p.add_argument("--no-plot", action="store_true", help="Skip the map-correlation heatmap.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = _config()
    suffix = cfg.EXCLUDED_RUNS_SUFFIX if args.exclude_broken_runs else ""
    fam, ref = cfg.CLUSTERS[args.family], cfg.CLUSTERS[args.reference_family]
    family_dir = Path(args.results_root) / (fam["derivatives_subdir"] + suffix)
    ref_group_dir = (Path(args.reference_group_dir) if args.reference_group_dir else
                     Path(args.results_root) / (ref["derivatives_subdir"] + suffix) / "group_level")

    pairs, per, r_mat = compare(_DATASET / fam["assignments_csv"], family_dir / "group_level",
                                _DATASET / ref["assignments_csv"], ref_group_dir,
                                ref_ids=args.reference_ids, ids=args.ids,
                                all_pairs=args.all_pairs)

    prefix = Path(args.out_prefix) if args.out_prefix else family_dir / f"compare_{args.reference_family}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for table, tag in ((pairs, ""), (per, "_per_cluster")):
        table.to_csv(f"{prefix}{tag}.csv", index=False)
        Path(f"{prefix}{tag}.md").write_text(to_markdown(table))
    print(to_markdown(per))
    written = [f"{prefix}.csv", f"{prefix}_per_cluster.csv"]
    if not args.no_plot and r_mat.notna().any().any():
        plot_map_r(r_mat, f"{prefix}_map_r.png", args.family, args.reference_family)
        written.append(f"{prefix}_map_r.png")
    print("Wrote " + ", ".join(written) + " (+ .md)")

    ident = pairs[pairs["relation"] == "identical"] if len(pairs) else pairs
    bad = ident[ident["map_r"].notna() & (ident["map_r"] < 0.999)] if len(ident) else ident
    if len(bad):
        print(f"WARNING: {len(bad)} identical-clip pair(s) with map_r < 0.999 — the runs differ: "
              + ", ".join(f"{int(a)}~{int(b)}" for a, b in zip(bad["cluster_id"], bad["ref_id"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
