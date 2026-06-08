"""
Generate one collage image per cluster, with a frame from the last 2 seconds
of each video in the cluster.
"""
import os
import json
import argparse
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
         "-frames:v", "1", "-q:v", "2", out_path,
         "-loglevel", "error"],
        check=True,
    )


def grab_frame(video_path, rng):
    """
    Grab a random frame from the last 2 seconds of the video.
    Returns a PIL Image, or None on failure.
    """
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
            target_frame = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame)
        else:
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
    """
    Arrange frames into a roughly-square grid collage.
    `labels` is a list of short strings drawn under each thumbnail.
    """
    n = len(frames)
    if n == 0:
        return None
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)

    tw, th = thumb_size
    cell_w = tw + pad
    cell_h = th + pad + label_h
    W = cols * cell_w + pad
    H = rows * cell_h + pad

    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    for i, (frame, label) in enumerate(zip(frames, labels)):
        r, c = divmod(i, cols)
        x = pad + c * cell_w
        y = pad + r * cell_h
        thumb = frame.copy()
        thumb.thumbnail(thumb_size, Image.LANCZOS)
        # center inside cell
        tx = x + (tw - thumb.width) // 2
        ty = y + (th - thumb.height) // 2
        canvas.paste(thumb, (tx, ty))
        # label below
        draw.text((x + 2, y + th + 2), label, fill=(220, 220, 220), font=font)

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters_json", type=str, required=True,
                        help="Path to clusters.json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to write per-cluster collages")
    parser.add_argument("--video_root", type=str, default="spacetop_embeddings_v2",
                        help="prefix prepended to relative video paths")
    parser.add_argument("--thumb_w", type=int, default=240)
    parser.add_argument("--thumb_h", type=int, default=135)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not HAS_CV2:
        print("opencv-python not found — falling back to ffmpeg/ffprobe.")

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with open(args.clusters_json, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    print(f"Loaded {len(clusters)} clusters from {args.clusters_json}")

    for cluster in clusters:
        cid = cluster["cluster_id"]
        vids = cluster["video_ids"]
        print(f"[cluster {cid}] {len(vids)} videos")

        frames, labels = [], []
        for vid in vids:
            path = os.path.join(args.video_root, vid) if args.video_root else vid
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

        collage = make_collage(
            frames, labels, thumb_size=(args.thumb_w, args.thumb_h))
        out_path = os.path.join(
            args.output_dir, f"cluster_{cid:03d}_n{len(frames)}.jpg")
        collage.save(out_path, quality=88)
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()