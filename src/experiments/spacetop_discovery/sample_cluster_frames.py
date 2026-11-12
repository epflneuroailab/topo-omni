"""Representative frames of a discovered SpaceTop cluster, for composing a figure panel.

One frame per clip, from clips drawn ROUND-ROBIN across the cluster's movies and spread
evenly through each movie's clips, so the sample shows the range of what the cluster holds
rather than one scene repeated. (``make_cluster_collages`` shows every clip as a small
thumbnail for labelling; this writes a few full-resolution frames for a figure.)

Writes, per cluster, under ``<output_dir>/cluster_frames/cluster_<id>/``:
  * ``<movie>_chunk_<NNNN>.png`` -- the middle frame of each sampled clip, native size
  * ``cluster_<id>_frames_grid.png`` -- all of them in one labelled grid, for picking

    python -m src.experiments.spacetop_discovery.sample_cluster_frames \\
        --model_name topo-omni-tiled32-4608 --run_tag disjoint2s_min60 --clusters 10,7
"""
import argparse
import collections
import os
import re
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.spacetop_discovery.common import (
    add_run_args, read_json, resolve_clip, resolve_run_args,
)

CHUNK = re.compile(r"^(?P<movie>.*)_chunk_(?P<n>\d+)\.mp4$")
CONTENT = re.compile(r"content-([A-Za-z0-9]+)")


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=noprint_wrappers=1:nokey=1", path],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def sample_clips(video_ids, n):
    """Round-robin across movies; within a movie, evenly spaced through its clips."""
    by_movie = collections.defaultdict(list)
    for v in video_ids:
        m = CHUNK.match(os.path.basename(v))
        by_movie[m.group("movie")].append((int(m.group("n")), v))
    movies = sorted(by_movie, key=lambda k: -len(by_movie[k]))
    quota = {m: 0 for m in movies}
    left = min(n, len(video_ids))
    while left:                                   # equal shares, capped by availability
        for m in movies:
            if left and quota[m] < len(by_movie[m]):
                quota[m] += 1
                left -= 1
    picks = []
    for m in movies:
        clips = sorted(by_movie[m])
        idx = np.unique(np.linspace(0, len(clips) - 1, quota[m]).round().astype(int))
        picks += [(m, clips[i][0], clips[i][1]) for i in idx]
    # interleave movies so any prefix of the grid is already mixed
    order = sorted(range(len(picks)),
                   key=lambda i: (sum(1 for j in range(i) if picks[j][0] == picks[i][0]),
                                  movies.index(picks[i][0])))
    return [picks[i] for i in order]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_run_args(ap)
    ap.add_argument("--clusters", required=True, help="comma-separated cluster ids")
    ap.add_argument("--n", type=int, default=24, help="clips (frames) per cluster")
    ap.add_argument("--cols", type=int, default=6, help="columns in the preview grid")
    ap.add_argument("--clusters_json", default=None,
                    help="Default: <output_dir>/clusters_tvals.json")
    args = resolve_run_args(ap.parse_args())

    out = args.output_dir
    clusters = {c["cluster_id"]: c["video_ids"]
                for c in read_json(args.clusters_json or os.path.join(out, "clusters_tvals.json"))}
    for cid in [int(c) for c in args.clusters.split(",")]:
        d = os.path.join(out, "cluster_frames", f"cluster_{cid}")
        os.makedirs(d, exist_ok=True)
        picks = sample_clips(clusters[cid], args.n)
        frames = []
        for movie, chunk, vid in picks:
            src = resolve_clip(vid, args.spacetop_dir)
            dst = os.path.join(d, f"{movie}_chunk_{chunk:04d}.png")
            if not os.path.exists(dst):
                t = 0.5 * duration(src)          # middle of the 2 s clip
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", src,
                                "-frames:v", "1", dst], check=True)
            content = CONTENT.search(movie)
            frames.append((dst, f"{content.group(1) if content else movie}  "
                                f"{2 * chunk}-{2 * chunk + 2}s"))
        counts = collections.Counter(m for m, _, _ in picks)
        print(f"> cluster {cid}: {len(frames)} frames from {len(counts)} movies "
              f"({', '.join(f'{CONTENT.search(m).group(1)}={k}' for m, k in counts.items())}) -> {d}")

        rows = int(np.ceil(len(frames) / args.cols))
        fig, axes = plt.subplots(rows, args.cols, figsize=(3.2 * args.cols, 2.1 * rows), dpi=150)
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, (path, label) in zip(np.atleast_1d(axes).ravel(), frames):
            ax.imshow(plt.imread(path))
            ax.set_title(label, fontsize=8)
        fig.suptitle(f"Cluster {cid} (n={len(clusters[cid])} clips)", fontsize=11)
        fig.tight_layout()
        grid_path = os.path.join(d, f"cluster_{cid}_frames_grid.png")
        fig.savefig(grid_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  grid -> {grid_path}")


if __name__ == "__main__":
    main()
