"""Attach source-video and timing info to each merged cluster.

Each clip id encodes the movie it came from and its 2s slot within it
(``..._content-<movie>_chunk_<idx>.mp4``); this turns that back into
``{video_name, start_time, end_time}`` so the downstream fMRI contrasts can line a
cluster up against the scan timeline.

    python -m src.experiments.spacetop_discovery.create_cluster_manifest \
        --model_name topo-omni-tiled32-4608
"""
import argparse
import os
import shlex
import subprocess
from glob import glob

from tqdm import tqdm

from src.experiments.spacetop_discovery.common import (
    add_run_args, read_json, resolve_run_args, videos_dir, write_json,
)

CHUNK_SECONDS = 2


def get_video_duration_ffprobe(video_path):
    cmd = (f"ffprobe -v error -show_entries format=duration "
           f"-of default=noprint_wrappers=1:nokey=1 {shlex.quote(video_path)}")
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.decode()}")
    return float(result.stdout.decode().strip())


def parse_clip_id(clip_id):
    """``ses-01_run-01_order-01_content-idiots_chunk_0007.mp4`` -> ``('idiots', 7)``."""
    base = os.path.basename(clip_id)
    tail = base.split("-")[-1]                      # idiots_chunk_0007.mp4
    parts = tail.split("_")
    return parts[0], int(parts[2].replace(".mp4", ""))


def get_num_chunks(video_list):
    """Highest chunk index seen per movie, i.e. how far the clip set actually goes."""
    video_to_num_chunks = {}
    for video_path in video_list:
        video_name, chunk_num = parse_clip_id(video_path)
        video_to_num_chunks[video_name] = max(
            chunk_num, video_to_num_chunks.get(video_name, chunk_num))
    return video_to_num_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Add source-video and timing info to the merged clusters.")
    add_run_args(parser)
    parser.add_argument("--manifest_path", type=str, default=None,
                        help="Default: <output_dir>/clips_manifest.json")
    parser.add_argument("--merged_clusters", type=str, default=None,
                        help="Default: <output_dir>/merged_clusters_tvals.json")
    parser.add_argument("--save_name", type=str,
                        default="merged_clusters_tvals_with_video_info.json")
    args = resolve_run_args(parser.parse_args())

    out = args.output_dir
    manifest_path = args.manifest_path or os.path.join(out, "clips_manifest.json")
    merged_path = args.merged_clusters or os.path.join(out, "merged_clusters_tvals.json")

    video_lists = read_json(manifest_path)
    video_to_num_chunks = get_num_chunks(video_lists)
    merged_clusters = read_json(merged_path)

    videos = glob(os.path.join(videos_dir(args.spacetop_dir), "*.mp4"))
    video_durations = {}
    for video in tqdm(videos, desc="Probing movie durations"):
        video_id = os.path.basename(video).split("-")[-1].replace(".mp4", "")
        duration = get_video_duration_ffprobe(video)
        num_chunks = int(duration // CHUNK_SECONDS)
        video_durations[video_id] = {"duration": duration, "num_chunks": num_chunks}

        if video_id in video_to_num_chunks:
            assert video_to_num_chunks[video_id] <= num_chunks, (
                f"Video {video_id} has chunk index {video_to_num_chunks[video_id]} in the "
                f"manifest but only {num_chunks} chunks fit in {duration:.2f}s")

    for cluster in merged_clusters:
        cluster["video_info"] = {}
        for video_id in cluster["video_ids"]:
            video_name, chunk_num = parse_clip_id(video_id)
            cluster["video_info"][video_id] = {
                "video_name": video_name,
                "start_time": chunk_num * CHUNK_SECONDS,
                "end_time": (chunk_num + 1) * CHUNK_SECONDS,
            }

    write_json(video_durations, os.path.join(out, "video_durations.json"))
    save_path = os.path.join(out, args.save_name)
    write_json(merged_clusters, save_path)
    print(f"Wrote {save_path}")


if __name__ == "__main__":
    main()
