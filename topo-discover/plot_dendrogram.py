"""
Plot dendrogram and export clusters from cached cluster_with_early_stopping output.
"""
import os
import sys
import json
import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


def load_pickle(path):
    with open(path, 'rb') as f:
        return pkl.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Plot dendrogram and export clusters from cached tree')
    parser.add_argument('--vectors_path', type=str,
                        default='task-alignvideo/clustering_v2/video_clip_vectors.npy')
    parser.add_argument('--manifest_path', type=str,
                        default='task-alignvideo/clustering_v2/clips_manifest.json')
    parser.add_argument('--output_dir', type=str,
                        default='task-alignvideo/clustering_v2')
    parser.add_argument('--linkage_name', type=str, default='linkage_tree_tvals_min=10_max=500.pkl',
                        help='Filename of cached cluster_with_early_stopping output')
    parser.add_argument('--max_label_leaves', type=int, default=80,
                        help='Above this many leaves, dendrogram x-labels are hidden')
    args = parser.parse_args()

    # --- Load inputs ---
    embeddings = np.load(args.vectors_path)
    with open(args.manifest_path, 'r', encoding='utf-8') as f:
        clips_manifest = json.load(f)

    min_len = min(len(embeddings), len(clips_manifest))
    embeddings = embeddings[:min_len]
    file_names = clips_manifest[:min_len]
    n = len(file_names)
    print(f"Loaded {n} embeddings of dim {embeddings.shape[1]}")

    tree_path = os.path.join(args.output_dir, args.linkage_name)
    if not os.path.exists(tree_path):
        print(f"Error: cached tree not found at {tree_path}")
        sys.exit(1)
    clusters = load_pickle(tree_path)
    print(f"Loaded {len(clusters)} clusters from {tree_path}")

    # --- Export clusters to JSON + TXT ---
    clusters_json_path = os.path.join(args.output_dir, 'clusters_tvals.json')
    clusters_txt_path = os.path.join(args.output_dir, 'clusters_tvals.txt')

    clusters_data = [
        {"cluster_id": i, "size": len(c), "video_ids": list(c)}
        for i, c in enumerate(clusters)
    ]
    with open(clusters_json_path, 'w', encoding='utf-8') as f:
        json.dump(clusters_data, f, indent=2)
    print(f"Wrote {clusters_json_path}")

    with open(clusters_txt_path, 'w', encoding='utf-8') as f:
        for i, c in enumerate(clusters):
            f.write(f"# Cluster {i} (n={len(c)})\n")
            for vid in c:
                f.write(f"{vid}\n")
            f.write("\n")
    print(f"Wrote {clusters_txt_path}")

    # --- Rebuild linkage matrix for dendrogram ---
    # Must match what cluster_with_early_stopping used
    print("Computing linkage matrix...")
    Z = linkage(embeddings, method='ward', metric='euclidean')

    # Map each video_id to its leaf index in the embedding order
    vid_to_idx = {vid: i for i, vid in enumerate(file_names)}

    # Determine the color for each cluster — assign each early-stopped cluster
    # a distinct color so it's visible on the dendrogram
    cmap = plt.get_cmap('tab20')
    leaf_color = {}  # leaf index -> hex color
    for i, c in enumerate(clusters):
        color = cmap(i % 20)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        for vid in c:
            if vid in vid_to_idx:
                leaf_color[vid_to_idx[vid]] = hex_color

    # link_color_func: color a link by the (shared) color of its descendant leaves,
    # otherwise gray
    def get_descendant_leaves(node_id):
        if node_id < n:
            return [node_id]
        row = int(node_id - n)
        return (get_descendant_leaves(int(Z[row, 0])) +
                get_descendant_leaves(int(Z[row, 1])))

    link_colors = {}
    for i in range(Z.shape[0]):
        node_id = n + i
        leaves = get_descendant_leaves(node_id)
        colors_in_subtree = {leaf_color.get(l, '#bbbbbb') for l in leaves}
        if len(colors_in_subtree) == 1:
            link_colors[node_id] = colors_in_subtree.pop()
        else:
            link_colors[node_id] = '#bbbbbb'

    # --- Plot dendrogram ---
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

    dendro_path = os.path.join(args.output_dir, 'dendrogram_tvals.png')
    plt.savefig(dendro_path, dpi=200, bbox_inches='tight')
    plt.savefig(dendro_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Wrote {dendro_path} (and .pdf)")
    plt.close()


if __name__ == "__main__":
    main()