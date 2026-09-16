"""Gather the per-movie .npz embeddings into one matrix + clip manifest.

Writes ``video_clip_vectors.npy`` ``(n_clips, 2048)`` and ``clips_manifest.json``
(the clip path per row) into the run's discovery directory -- the two inputs
``agglomerative_early_stopping`` reads.

    python -m src.experiments.spacetop_discovery.compile_embeddings \
        --model_name topo-omni-tiled32-4608
"""
import argparse
import os
from glob import glob

import numpy as np

from src.experiments.spacetop_discovery.common import (
    add_run_args, clips_dir, resolve_run_args, write_json,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compile per-movie clip embeddings into one matrix + manifest.")
    add_run_args(parser)
    parser.add_argument("--embeddings_dir", type=str, default=None,
                        help="Default: $SPACETOP_DIR/spacetop_embeddings_v2")
    args = resolve_run_args(parser.parse_args())

    dirpath = args.embeddings_dir or clips_dir(args.spacetop_dir)

    path_embeddings = sorted(glob(os.path.join(dirpath, "*nemotron_embeddings.npz")))
    print(f"Found {len(path_embeddings)} embedding files in {dirpath}.")

    out_manifest = os.path.join(args.output_dir, "clips_manifest.json")
    out_embeddings = os.path.join(args.output_dir, "video_clip_vectors.npy")

    manifest, all_embeddings = [], []
    for emb_path in path_embeddings:
        data = np.load(emb_path)
        all_embeddings.extend(data["embeddings"])   # (num_clips, embedding_dim)
        manifest.extend(data["chunk_paths"])

    all_embeddings = np.stack(all_embeddings, axis=0)
    print(f"Compiled embeddings shape: {all_embeddings.shape}")

    np.save(out_embeddings, all_embeddings)
    write_json([str(m) for m in manifest], out_manifest)
    print(f"Wrote {out_embeddings} and {out_manifest}")


if __name__ == "__main__":
    main()
