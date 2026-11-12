"""Regress movie identity out of the clip embeddings before the Ward tree.

Discovery on the raw nemotron vectors recovers "which movie is this" more than
"what is in this clip": AMI(cluster, movie) = 0.73 on disjoint 2s clips, against a
size-matched null of ~0.00. Each movie is internally homogeneous -- one speaker,
one palette, one score, one encoder pass -- so its clips sit together whatever
they depict, and cutting the clip overlap barely moved the number (0.80 -> 0.73).

Projecting out the one-hot movie design matrix ``M`` is exactly within-movie
centering::

    X_resid = X - M (M^T M)^-1 M^T X

leaving each clip as its deviation from its own movie's mean. Ward runs on
Euclidean distance, which is translation invariant, so the global mean is added
back only to keep the vectors in a familiar range (and it *does* matter if you
pass --renormalize).

CAVEAT, and it is not a small one: this cannot separate "movie identity" from
"content that only ever occurs in one movie". A movie that is wall-to-wall space
footage has its space signal removed along with its identity, so genuine
single-movie categories will weaken. What survives is structure that recurs
ACROSS movies -- which is the point, but it is a different question than the raw
run asks, not a strictly better answer to the same one.

Writes a sibling run directory that shares the source run's cortical sheets by
symlink -- the sheets depend only on the clip and the checkpoint, never on how the
embeddings were transformed, so re-extracting them would be pure waste.

    python -m src.experiments.spacetop_discovery.residualize_movie \
        --model_name topo-omni-tiled32-4608 \
        --source_tag disjoint2s --run_tag disjoint2s_resid
"""
import argparse
import os
import re

import numpy as np

from src.experiments.spacetop_discovery.common import (
    add_run_args, discovery_dir, read_json, resolve_run_args, write_json,
)

# `ses-01_run-01_order-01_content-idiots_chunk_0007.mp4` -> the movie it was cut from.
# Keyed on the WHOLE stem, not the `content-<name>` tail: two runs can share a content
# name, and collapsing them would centre two different movies on one mean.
_CHUNK_RE = re.compile(r"^(.*)_chunk_\d+\.mp4$")


def movie_of(clip_path):
    base = os.path.basename(clip_path)
    m = _CHUNK_RE.match(base)
    return m.group(1) if m else base


def residualize(vectors, movies, renormalize=False):
    """Within-movie centering, i.e. projecting out the one-hot movie design matrix.

    Returns ``(residuals, n_movies)``. Movies contributing a single clip come out as
    exactly zero (their mean is themselves), so they are reported by the caller.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    out = np.empty_like(vectors)
    global_mean = vectors.mean(axis=0)

    uniq = sorted(set(movies))
    movies = np.asarray(movies)
    for name in uniq:
        idx = np.flatnonzero(movies == name)
        out[idx] = vectors[idx] - vectors[idx].mean(axis=0)

    out += global_mean

    if renormalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out /= norms

    return out.astype(np.float32), len(uniq)


def main():
    parser = argparse.ArgumentParser(
        description="Project movie identity out of the clip embeddings.")
    add_run_args(parser)
    parser.add_argument("--source_tag", type=str, default=None,
                        help="run_tag of the run whose vectors/manifest/sheets to reuse.")
    parser.add_argument("--source_dir", type=str, default=None,
                        help="Explicit source directory; overrides --source_tag.")
    parser.add_argument("--renormalize", action="store_true",
                        help="L2-normalise the residuals. Off by default: the raw "
                             "vectors are unit-norm, but rescaling after centering "
                             "re-weights clips by how far they sit from their movie.")
    parser.add_argument("--link_sheets", action="store_true", default=True,
                        help="Symlink the source run's cortical_sheets (default).")
    args = resolve_run_args(parser.parse_args())

    src = args.source_dir or discovery_dir(args.model_name, args.source_tag)
    dst = args.output_dir
    if os.path.abspath(src) == os.path.abspath(dst):
        parser.error("source and output directory are the same; pass a different --run_tag")

    vectors = np.load(os.path.join(src, "video_clip_vectors.npy"))
    manifest = read_json(os.path.join(src, "clips_manifest.json"))
    n = min(len(vectors), len(manifest))
    vectors, manifest = vectors[:n], manifest[:n]

    movies = [movie_of(p) for p in manifest]
    resid, n_movies = residualize(vectors, movies, renormalize=args.renormalize)

    counts = {m: movies.count(m) for m in set(movies)}
    singletons = [m for m, c in counts.items() if c < 2]

    print(f"{n} clips from {n_movies} movies")
    print(f"  mean |x| before {np.linalg.norm(vectors, axis=1).mean():.4f} "
          f"-> after {np.linalg.norm(resid, axis=1).mean():.4f}")
    var_before = float(vectors.var(axis=0).sum())
    var_after = float(resid.var(axis=0).sum())
    print(f"  total variance {var_before:.4f} -> {var_after:.4f} "
          f"({100 * var_after / var_before:.0f}% retained; the rest WAS movie identity)")
    if singletons:
        print(f"  ! {len(singletons)} movie(s) contribute one clip and residualise to "
              f"the global mean exactly: {singletons[:5]}")

    np.save(os.path.join(dst, "video_clip_vectors.npy"), resid)
    write_json(manifest, os.path.join(dst, "clips_manifest.json"))

    if args.link_sheets:
        link = os.path.join(dst, "cortical_sheets")
        target = os.path.join(src, "cortical_sheets")
        if not os.path.exists(link) and os.path.isdir(target):
            os.symlink(target, link)
            print(f"  linked cortical_sheets -> {target}")

    print(f"Wrote residualised run directory: {dst}")


if __name__ == "__main__":
    main()
