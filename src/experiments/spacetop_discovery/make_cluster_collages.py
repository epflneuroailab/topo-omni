"""One collage per cluster: a frame from the last 2s of each clip in it.

This is the labelling step -- you read the collages and write the merge map that
``merge_clusters`` consumes.

    python -m src.experiments.spacetop_discovery.make_cluster_collages \
        --model_name topo-omni-tiled32-4608
"""
import argparse
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.experiments.spacetop_discovery.common import (
    add_run_args, read_json, resolve_clip, resolve_run_args,
)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def get_video_duration_ffprobe(video_path):
    """Return video duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def extract_frame_ffmpeg(video_path, timestamp, out_path):
    """Extract a single frame at `timestamp` seconds via ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", video_path,
         "-frames:v", "1", "-q:v", "2", out_path, "-loglevel", "error"],
        check=True,
    )


def grab_frame(video_path, rng):
    """Random frame from the last 2 seconds. PIL Image, or None on failure."""
    try:
        if HAS_CV2:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n_frames <= 0:
                cap.release()
                return None
            duration = n_frames / fps
            t_start = max(0.0, duration - 2.0)
            t = rng.uniform(t_start, duration - 1.0 / fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                return None
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        duration = get_video_duration_ffprobe(video_path)
        t_start = max(0.0, duration - 2.0)
        t = rng.uniform(t_start, max(t_start, duration - 0.05))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            extract_frame_ffmpeg(video_path, t, tmp_path)
            return Image.open(tmp_path).convert("RGB").copy()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        print(f"  ! Failed to extract frame from {video_path}: {e}")
        return None


def make_collage(frames, labels, thumb_size=(240, 135), pad=8, label_h=18):
    """Arrange frames into a roughly-square grid, each captioned with a short label."""
    n = len(frames)
    if n == 0:
        return None
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)

    tw, th = thumb_size
    cell_w, cell_h = tw + pad, th + pad + label_h
    canvas = Image.new("RGB", (cols * cell_w + pad, rows * cell_h + pad), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    for i, (frame, label) in enumerate(zip(frames, labels)):
        r, c = divmod(i, cols)
        x, y = pad + c * cell_w, pad + r * cell_h
        thumb = frame.copy()
        thumb.thumbnail(thumb_size, Image.LANCZOS)
        canvas.paste(thumb, (x + (tw - thumb.width) // 2, y + (th - thumb.height) // 2))
        draw.text((x + 2, y + th + 2), label, fill=(220, 220, 220), font=font)

    return canvas


def main():
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    parser.add_argument("--clusters_json", type=str, default=None,
                        help="Default: <output_dir>/clusters_tvals.json")
    parser.add_argument("--previews_dir", type=str, default=None,
                        help="Default: <output_dir>/cluster_previews")
    parser.add_argument("--max_per_cluster", type=int, default=100,
                        help="Cap the frames per collage; a 500-clip cluster otherwise "
                             "renders a wall no one can read.")
    parser.add_argument("--thumb_w", type=int, default=240)
    parser.add_argument("--thumb_h", type=int, default=135)
    parser.add_argument("--seed", type=int, default=0)
    args = resolve_run_args(parser.parse_args())

    if not HAS_CV2:
        print("opencv-python not found - falling back to ffmpeg/ffprobe.")

    clusters_json = args.clusters_json or os.path.join(args.output_dir, "clusters_tvals.json")
    previews_dir = args.previews_dir or os.path.join(args.output_dir, "cluster_previews")
    os.makedirs(previews_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    clusters = read_json(clusters_json)
    print(f"Loaded {len(clusters)} clusters from {clusters_json}")

    for cluster in clusters:
        cid = cluster["cluster_id"]
        vids = cluster["video_ids"]
        if args.max_per_cluster and len(vids) > args.max_per_cluster:
            pick = rng.choice(len(vids), size=args.max_per_cluster, replace=False)
            vids = [vids[i] for i in sorted(pick)]
            print(f"[cluster {cid}] {cluster['size']} videos "
                  f"(showing {len(vids)})")
        else:
            print(f"[cluster {cid}] {len(vids)} videos")

        frames, labels = [], []
        for vid in vids:
            path = resolve_clip(vid, args.spacetop_dir)
            if not os.path.exists(path):
                print(f"  ! Missing: {path}")
                continue
            frame = grab_frame(path, rng)
            if frame is not None:
                frames.append(frame)
                labels.append(Path(vid).stem[:28])

        if not frames:
            print(f"  ! No frames extracted for cluster {cid}, skipping")
            continue

        collage = make_collage(frames, labels, thumb_size=(args.thumb_w, args.thumb_h))
        out_path = os.path.join(previews_dir, f"cluster_{cid:03d}_n{len(frames)}.jpg")
        collage.save(out_path, quality=88)
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
