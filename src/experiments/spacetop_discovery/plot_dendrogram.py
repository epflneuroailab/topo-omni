"""Export the early-stopped clusters and draw the dendrogram they cut.

Reads the pickle written by ``agglomerative_early_stopping`` and turns it into the
``clusters_tvals.json`` every later step consumes, plus a coloured dendrogram where
each kept cluster gets its own colour and everything above the cut is grey.

    python -m src.experiments.spacetop_discovery.plot_dendrogram \
        --model_name topo-omni-tiled32-4608
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

from src.experiments.spacetop_discovery.common import (
    add_run_args, read_json, read_pickle, resolve_run_args, write_json,
)


def main():
    parser = argparse.ArgumentParser(
        description='Plot dendrogram and export clusters from the cached tree')
    add_run_args(parser)
    parser.add_argument('--vectors_path', type=str, default=None)
    parser.add_argument('--manifest_path', type=str, default=None)
    parser.add_argument('--linkage_name', type=str,
                        default='linkage_tree_tvals_min=10_max=500.pkl',
                        help='Filename of the cached cluster_with_early_stopping output')
    parser.add_argument('--max_label_leaves', type=int, default=80,
                        help='Above this many leaves, dendrogram x-labels are hidden')
    args = resolve_run_args(parser.parse_args())

    out = args.output_dir
    vectors_path = args.vectors_path or os.path.join(out, 'video_clip_vectors.npy')
    manifest_path = args.manifest_path or os.path.join(out, 'clips_manifest.json')

    embeddings = np.load(vectors_path)
    clips_manifest = read_json(manifest_path)

    min_len = min(len(embeddings), len(clips_manifest))
    embeddings = embeddings[:min_len]
    clips_manifest = clips_manifest[:min_len]

    # The walk clustered only the clips whose cortical sheet came out clean, and wrote
    # that list out. Rebuild the linkage over exactly those, or the leaf indices here
    # address a different tree than the one the clusters were cut from.
    used_path = os.path.join(out, 'clips_used.json')
    if os.path.exists(used_path):
        used = read_json(used_path)
        pos = {vid: i for i, vid in enumerate(clips_manifest)}
        keep = [pos[v] for v in used if v in pos]
        if len(keep) != len(clips_manifest):
            print(f"Restricting to the {len(keep)} clips the walk used "
                  f"(of {len(clips_manifest)} in the manifest)")
        embeddings = embeddings[keep]
        file_names = [clips_manifest[i] for i in keep]
    else:
        print(f"Note: no {used_path}; assuming the walk used the whole manifest.")
        file_names = clips_manifest

    n = len(file_names)
    print(f"Loaded {n} embeddings of dim {embeddings.shape[1]}")

    tree_path = os.path.join(out, args.linkage_name)
    if not os.path.exists(tree_path):
        print(f"Error: cached tree not found at {tree_path}")
        sys.exit(1)
    clusters = read_pickle(tree_path)
    print(f"Loaded {len(clusters)} clusters from {tree_path}")

    # --- Export clusters to JSON + TXT ---
    clusters_json_path = os.path.join(out, 'clusters_tvals.json')
    clusters_txt_path = os.path.join(out, 'clusters_tvals.txt')

    clusters_data = [
        {"cluster_id": i, "size": len(c), "video_ids": list(c)}
        for i, c in enumerate(clusters)
    ]
    write_json(clusters_data, clusters_json_path, indent=2)
    print(f"Wrote {clusters_json_path}")

    with open(clusters_txt_path, 'w', encoding='utf-8') as f:
        for i, c in enumerate(clusters):
            f.write(f"# Cluster {i} (n={len(c)})\n")
            for vid in c:
                f.write(f"{vid}\n")
            f.write("\n")
    print(f"Wrote {clusters_txt_path}")

    # --- Rebuild linkage matrix for dendrogram (must match the walk's) ---
    print("Computing linkage matrix...")
    Z = linkage(embeddings, method='ward', metric='euclidean')

    vid_to_idx = {vid: i for i, vid in enumerate(file_names)}

    cmap = plt.get_cmap('tab20')
    leaf_color = {}
    for i, c in enumerate(clusters):
        color = cmap(i % 20)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        for vid in c:
            if vid in vid_to_idx:
                leaf_color[vid_to_idx[vid]] = hex_color

    # Colour a link by the shared colour of its descendant leaves, else grey. Leaves
    # per node bottom-up: Z's rows always merge already-built nodes.
    node_leaves = [[i] for i in range(n)]
    for row in range(Z.shape[0]):
        node_leaves.append(node_leaves[int(Z[row, 0])] + node_leaves[int(Z[row, 1])])

    link_colors = {}
    for i in range(Z.shape[0]):
        node_id = n + i
        colors_in_subtree = {leaf_color.get(l, '#bbbbbb') for l in node_leaves[node_id]}
        link_colors[node_id] = colors_in_subtree.pop() if len(colors_in_subtree) == 1 \
            else '#bbbbbb'

    fig, ax = plt.subplots(figsize=(max(12, n * 0.15), 8))
    show_labels = n <= args.max_label_leaves
    dendrogram(
        Z,
        labels=file_names if show_labels else None,
        no_labels=not show_labels,
        leaf_rotation=90,
        leaf_font_size=8,
        link_color_func=lambda k: link_colors.get(k, '#bbbbbb'),
        ax=ax,
    )
    ax.set_title(
        f'Hierarchical clustering ({n} clips, {len(clusters)} early-stopped clusters)')
    ax.set_xlabel('Video clip')
    ax.set_ylabel('Ward distance')
    plt.tight_layout()

    dendro_path = os.path.join(out, 'dendrogram_tvals.png')
    plt.savefig(dendro_path, dpi=200, bbox_inches='tight')
    plt.savefig(dendro_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Wrote {dendro_path} (and .pdf)")
    plt.close()


if __name__ == "__main__":
    main()
