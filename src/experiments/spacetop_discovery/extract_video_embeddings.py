"""Cut the SpaceTop movies into 2s clips and embed each with Omni-Embed-Nemotron.

These embeddings are the *stimulus-side* description of the clips -- they seed the
Ward tree that ``agglomerative_early_stopping`` then walks. They come from a
separate off-the-shelf model on purpose: the hierarchy being cut must not itself be
a product of the checkpoint under test.

Slow (one model forward per 2s clip), so it shards: run ``--num-groups`` copies with
``--group-index`` 0..N-1. Already-extracted runs are skipped.

    python -m src.experiments.spacetop_discovery.extract_video_embeddings \
        --group-index 0 --num-groups 10
"""
import argparse
import functools
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

from qwen_omni_utils import process_mm_info

from src.experiments.spacetop_discovery.common import clips_dir, videos_dir

EMBED_MODEL = "nvidia/omni-embed-nemotron-3b"
EMBED_DIM = 2048


@functools.lru_cache(maxsize=None)
def _processor_source(model_name):
    """Where to load the embedding processor from -- a local dir, not the hub id.

    transformers 4.57's hub path lists a repo's additional chat templates WITH their
    ``.jinja`` extension and then appends ``.jinja`` again, so it tries to resolve
    ``additional_chat_templates/sentence_transformers.jinja.jinja``, gets ``None``
    back, and dies in ``open(None)`` with "expected str, bytes or os.PathLike object,
    not NoneType". The local-directory branch globs the template dir and uses each
    file's *stem*, so it is unaffected -- materialise the repo's small files and load
    from that directory. It is the same cache entry, so nothing is re-downloaded.
    """
    if os.path.isdir(model_name):
        return model_name
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(
            model_name,
            allow_patterns=["*.json", "*.jinja", "*.txt", "*.model",
                            "additional_chat_templates/*"],
        )
    except Exception as exc:
        print(f"! could not stage processor files locally ({exc}); using {model_name}")
        return model_name


@functools.lru_cache(maxsize=1)
def _video_encoder():
    """``(ffmpeg_binary, video_encoder_args)`` for the disjoint cut's re-encode.

    The conda env's ffmpeg 6.1 is built WITHOUT libx264 -- it carries only
    libopenh264, and `-preset` then fails argument parsing outright. imageio-ffmpeg
    ships a full static build that does have libx264, so prefer that; fall back to
    libopenh264 at a fixed bitrate on whatever is on PATH. (ffprobe still comes from
    PATH -- imageio-ffmpeg ships ffmpeg only.)
    """
    candidates = []
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    candidates.append("ffmpeg")

    for exe in candidates:
        try:
            out = subprocess.run([exe, "-hide_banner", "-encoders"],
                                 capture_output=True, text=True).stdout
        except Exception:
            continue
        if " libx264 " in out:
            return exe, ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    return "ffmpeg", ["-c:v", "libopenh264", "-b:v", "4M"]


class OmniEmbedExtractor:
    def __init__(self, model_name=EMBED_MODEL, device="cuda:0"):
        print(f"Loading embedding model: {model_name}")
        self.device = device

        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.model = self.model.to(device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            _processor_source(model_name), trust_remote_code=True)

    def split_video_by_time(self, video_path, chunk_duration=2.0, output_dir=None,
                            mode="disjoint"):
        """Cut `video_path` into clips with ffmpeg. Returns chunk metadata dicts.

        ``mode="disjoint"`` (default) tiles the movie: clip *N* covers
        ``[2N, 2N+2)``. One segment-muxer pass per movie, with a forced keyframe on
        every boundary so the cuts are frame-accurate -- stream copy would snap each
        cut to the nearest existing keyframe, which for a 2s clip can miss the
        boundary entirely. Costs a re-encode, but only 1 ffmpeg call per movie.

        ``mode="prefix"`` reproduces the original behaviour, in which `start_time`
        was never advanced so clip *N* was the movie's first ``2(N+1)`` seconds --
        clips that overlap almost completely and grow to minutes. Kept so the
        published run stays reproducible; it is not what you want for discovery,
        because near-identical consecutive prefixes make Ward cluster by movie
        identity rather than by content.
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = Path(output_dir)

        video_path = Path(video_path)

        probe_cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path),
        ]
        duration = float(subprocess.check_output(probe_cmd).decode().strip())

        if mode == "prefix":
            return self._split_prefix(video_path, duration, chunk_duration, output_dir)
        if mode != "disjoint":
            raise ValueError(f"unknown chunk mode {mode!r}")

        pattern = str(output_dir / f"{video_path.stem}_chunk_%04d.mp4")
        ffmpeg, venc = _video_encoder()
        cmd = [
            ffmpeg, '-y', '-i', str(video_path),
            '-map', '0:v:0', '-map', '0:a:0?',
            *venc,
            '-force_key_frames', f'expr:gte(t,n_forced*{chunk_duration})',
            '-c:a', 'aac', '-b:a', '128k',
            '-f', 'segment', '-segment_time', str(chunk_duration),
            '-reset_timestamps', '1',
            pattern,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg segment cut failed for {video_path.name} "
                f"(rc={proc.returncode}):\n{proc.stderr[-2000:]}")

        chunk_files = []
        for chunk_file in sorted(output_dir.glob(f"{video_path.stem}_chunk_*.mp4")):
            chunk_idx = int(chunk_file.stem.rsplit("_", 1)[1])
            start_time = chunk_idx * chunk_duration
            # The muxer emits a final short remainder; drop it so every clip the model
            # sees is the same length and `create_cluster_manifest`'s 2s arithmetic holds.
            if start_time + chunk_duration > duration + 1e-3:
                chunk_file.unlink()
                continue
            chunk_files.append({
                'path': chunk_file,
                'chunk_id': chunk_idx,
                'start_time': start_time,
                'end_time': start_time + chunk_duration,
                'duration': chunk_duration,
            })

        return chunk_files

    def _split_prefix(self, video_path, duration, chunk_duration, output_dir):
        """Legacy cumulative-prefix cutting -- see `split_video_by_time`."""
        chunk_files = []
        chunk_idx = 0
        start_time = 0.0
        accum_time = chunk_duration

        while accum_time < duration:
            end_time = min(start_time + accum_time, duration)
            chunk_file = output_dir / f"{video_path.stem}_chunk_{chunk_idx:04d}.mp4"

            cmd = [
                'ffmpeg', '-y', '-i', str(video_path),
                '-ss', str(start_time), '-t', str(accum_time),
                '-c', 'copy',           # stream copy: no re-encode
                str(chunk_file),
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            chunk_files.append({
                'path': chunk_file,
                'chunk_id': chunk_idx,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
            })

            chunk_idx += 1
            accum_time += chunk_duration

        return chunk_files

    def extract_embedding(self, video_path, use_audio=False, prompt=None):
        """L2-normalised last-token embedding of one clip. Returns ``(2048,)``."""
        content = []
        if prompt:
            content.append({"type": "text", "text": f"passage: {prompt}"})
        content.append({"type": "video", "video": str(video_path)})
        if use_audio:
            content.append({"type": "audio", "audio": str(video_path)})

        documents = [{"role": "user", "content": content}]

        documents_texts = self.processor.apply_chat_template(
            documents, add_generation_prompt=False, tokenize=False)

        audio, images, videos = process_mm_info(documents, use_audio_in_video=use_audio)

        videos_kwargs = {"min_pixels": 28 * 28, "max_pixels": 1280 * 28 * 28, "fps": 2.0}

        inputs = self.processor(
            text=documents_texts, audio=audio, images=images, videos=videos,
            videos_kwargs=videos_kwargs, return_tensors="pt", padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]     # (batch, seq_len, hidden)
            embeddings = hidden_states[:, -1, :]          # last token
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu().float().numpy()[0]

    def process_video_chunks(self, video_path, chunk_duration=2.0, use_audio=True,
                             output_dir=None, keep_chunks=False, prompt=None,
                             mode="disjoint"):
        """Split one movie, embed every chunk, and save the .npz + metadata."""
        video_path = Path(video_path)
        temp_dir = tempfile.mkdtemp() if not keep_chunks else output_dir

        print(f"Splitting video into {chunk_duration}s chunks...")
        chunks = self.split_video_by_time(video_path, chunk_duration, temp_dir, mode=mode)

        print(f"Extracting embeddings from {len(chunks)} chunks...")
        results = []
        for chunk_info in chunks:
            print(f"Processing chunk {chunk_info['chunk_id']}...", end='\r')

            embedding = self.extract_embedding(
                chunk_info['path'], use_audio=use_audio, prompt=prompt)

            results.append({
                'chunk_id': chunk_info['chunk_id'],
                'start_time': chunk_info['start_time'],
                'end_time': chunk_info['end_time'],
                'duration': chunk_info['duration'],
                'embedding': embedding,
                'embedding_dim': embedding.shape[0],
            })

            if not keep_chunks:
                chunk_info['path'].unlink()

        print(f"\nCompleted {len(results)} chunks")

        output_data = {
            'video_path': str(video_path),
            'num_chunks': len(results),
            'chunk_duration': chunk_duration,
            'chunk_mode': mode,
            'use_audio': use_audio,
            'model': EMBED_MODEL,
            'embedding_dim': EMBED_DIM,
            'chunks': results,
        }

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            embeddings_array = np.array([chunk['embedding'] for chunk in results])
            output_file = output_dir / f"{video_path.stem}_nemotron_embeddings.npz"

            chunk_paths = [str(chunks[i]['path']) for i in range(len(results))] \
                if keep_chunks else []

            np.savez_compressed(
                output_file,
                embeddings=embeddings_array,
                timestamps=np.array([[c['start_time'], c['end_time']] for c in results]),
                chunk_ids=np.array([c['chunk_id'] for c in results]),
                chunk_paths=np.array(chunk_paths) if chunk_paths else None,
                video_name=video_path.stem,
            )

            metadata_file = output_dir / f"{video_path.stem}_metadata.json"
            json_data = output_data.copy()
            for i, chunk in enumerate(json_data['chunks']):
                chunk['embedding'] = f"Saved in {output_file.name}"
                if keep_chunks and i < len(chunk_paths):
                    chunk['chunk_path'] = chunk_paths[i]

            with open(metadata_file, 'w') as f:
                json.dump(json_data, f, indent=2)

            print(f"Saved embeddings to {output_file}")
            print(f"Shape: {embeddings_array.shape}")

        return output_data


def main():
    parser = argparse.ArgumentParser(description='Extract video clip embeddings')
    parser.add_argument('--group-index', type=int, default=0,
                        help='Index of the video shard to process')
    parser.add_argument('--num-groups', type=int, default=10,
                        help='Number of shards the movie list is split into')
    parser.add_argument('--video-dir', type=str, default=None,
                        help='Default: $SPACETOP_DIR/task-alignvideo')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Default: $SPACETOP_DIR/spacetop_embeddings_v2')
    parser.add_argument('--spacetop_dir', type=str, default=None)
    parser.add_argument('--chunk-duration', type=float, default=2.0)
    parser.add_argument('--chunk-mode', type=str, default='disjoint',
                        choices=['disjoint', 'prefix'],
                        help="'disjoint' tiles the movie into non-overlapping clips. "
                             "'prefix' reproduces the original run, where clip N was the "
                             "movie's first 2(N+1) seconds.")
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    video_dir = Path(args.video_dir or videos_dir(args.spacetop_dir))
    output_base = Path(args.output_dir or clips_dir(args.spacetop_dir))

    video_files = sorted(video_dir.glob("*.mp4"))
    video_files = [video_files[i::args.num_groups] for i in range(args.num_groups)][args.group_index]

    print(f"Shard {args.group_index}/{args.num_groups}: {len(video_files)} videos "
          f"from {video_dir} -> {output_base} [{args.chunk_mode} {args.chunk_duration}s]")

    extractor = OmniEmbedExtractor(model_name=EMBED_MODEL, device=args.device)

    for video_file in video_files:
        done = output_base / f"{video_file.stem}_nemotron_embeddings.npz"
        if done.exists():
            print(f"= Skipping {video_file.name} (already extracted)")
            continue
        print(f"\nProcessing: {video_file.name}")
        try:
            extractor.process_video_chunks(
                video_path=video_file,
                chunk_duration=args.chunk_duration,
                mode=args.chunk_mode,
                use_audio=True,
                output_dir=output_base,
                keep_chunks=True,       # the clips are the input to every later step
            )
            print(f"+ Completed {video_file.name}")
        except Exception as e:
            print(f"! Error processing {video_file.name}: {e}")
            continue


if __name__ == "__main__":
    main()
