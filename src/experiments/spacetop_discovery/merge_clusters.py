"""Merge the raw early-stopped clusters into labelled ones.

Early stopping over-splits: a face-selective region shows up as half a dozen
adjacent leaves. Which leaves belong together is a judgement call made by looking
at ``make_cluster_collages`` output, so the grouping lives in a JSON *merge map*
rather than in this file -- the old version hardcoded one run's cluster ids, which
silently mislabels every other run's tree.

The map is ``{"<label>": [<cluster_id>, ...], ...}``::

    {"space": [0, 1, 2, 3, 4], "faces_1": [33, 34, 35], "paintings": [51, 52, 53]}

    # write a starter map (one label per raw cluster) to fill in by hand
    python -m src.experiments.spacetop_discovery.merge_clusters \
        --model_name topo-omni-tiled32-4608 --init_map

    # then merge
    python -m src.experiments.spacetop_discovery.merge_clusters \
        --model_name topo-omni-tiled32-4608 --merge_map <path>
"""
import argparse
import os
import sys

from src.experiments.spacetop_discovery.common import (
    add_run_args, read_json, resolve_run_args, write_json,
)


def main():
    parser = argparse.ArgumentParser(
        description="Merge raw discovery clusters into labelled clusters.")
    add_run_args(parser)
    parser.add_argument("--clusters_json", type=str, default=None,
                        help="Default: <output_dir>/clusters_tvals.json")
    parser.add_argument("--merge_map", type=str, default=None,
                        help="JSON {label: [cluster_id, ...]}. "
                             "Default: <output_dir>/merge_map.json")
    parser.add_argument("--save_name", type=str, default="merged_clusters_tvals.json")
    parser.add_argument("--init_map", action="store_true",
                        help="Write a starter merge map (one label per raw cluster, "
                             "sizes in the label) and exit. Never overwrites.")
    args = resolve_run_args(parser.parse_args())

    out = args.output_dir
    clusters_path = args.clusters_json or os.path.join(out, "clusters_tvals.json")
    merge_map_path = args.merge_map or os.path.join(out, "merge_map.json")

    original_clusters = read_json(clusters_path)

    if args.init_map:
        if os.path.exists(merge_map_path):
            print(f"Refusing to overwrite existing {merge_map_path}")
            sys.exit(1)
        template = {f"cluster_{row['cluster_id']:03d}_n{row['size']}": [row["cluster_id"]]
                    for row in original_clusters}
        write_json(template, merge_map_path)
        print(f"Wrote starter merge map with {len(template)} entries to {merge_map_path}\n"
              f"Rename the keys and group the ids by hand, then re-run without --init_map.")
        return

    if not os.path.exists(merge_map_path):
        print(f"Error: no merge map at {merge_map_path}.\n"
              f"Look at the collages, then run with --init_map to get a starter file.")
        sys.exit(1)

    clusters_to_merge = {label: set(ids) for label, ids in read_json(merge_map_path).items()}

    known = {row["cluster_id"] for row in original_clusters}
    mapped = {cid for ids in clusters_to_merge.values() for cid in ids}
    if mapped - known:
        print(f"Error: merge map references clusters that do not exist: "
              f"{sorted(mapped - known)}")
        sys.exit(1)
    if known - mapped:
        print(f"Note: {len(known - mapped)} raw clusters are in no merge group and are "
              f"dropped: {sorted(known - mapped)}")

    merged_clusters = []
    for idx, (cluster_label, cluster_ids) in enumerate(clusters_to_merge.items()):
        video_ids = []
        for row in original_clusters:
            if row["cluster_id"] in cluster_ids:
                video_ids.extend(row["video_ids"])

        merged_clusters.append({
            "cluster_id": idx + 1,
            "cluster_label": cluster_label,
            "size": len(video_ids),
            "video_ids": video_ids,
        })

    save_path = os.path.join(out, args.save_name)
    write_json(merged_clusters, save_path)
    print(f"Merged cluster assignments saved to '{save_path}'")


if __name__ == "__main__":
    main()
