"""Discover video-driven clusters by walking a Ward dendrogram with early stopping.

The clip embeddings (NVIDIA Omni-Embed-Nemotron, see ``extract_video_embeddings``)
give a *stimulus-side* hierarchy of the 2s SpaceTop clips. That hierarchy alone has
no privileged cut. So we walk it top-down and keep splitting a node only while
splitting makes the resulting clip set *more selective on the model's cortical
sheet*: each candidate cluster is scored by the median Welch t of its clips'
unified sheets against every other clip's, and a node is kept whole once both
children score below it.

That means the tree depends on the checkpoint -- the sheets are the model's --
so results live under ``$SAVE_DIR/<model_name>/spacetop_discovery``.

Run from the repo root::

    python -m src.experiments.spacetop_discovery.agglomerative_early_stopping \
        --model_name topo-omni-tiled32-4608 \
        --checkpoint artifacts/ckpts/topo-omni-tiled32-4608

``TOPO_SHEET_LAYOUT`` must match the checkpoint (``tiled32`` for the 800x576 runs)
or a random front-end adaptor is silently applied -- it is read at import time, so
export it in the environment rather than relying on ``.env``.
"""
from typing import List

import argparse
import os
import subprocess
import sys

import numpy as np
import scipy.stats as stats
import torch
from scipy.cluster.hierarchy import linkage
from tqdm import tqdm

from qwen_omni_utils import process_mm_info

from src.core.model_loading import load_topo_omni
from src.experiments.spacetop_discovery.common import (
    add_run_args, read_json, resolve_clip, resolve_run_args, write_json, write_pickle,
)

USE_AUDIO_IN_VIDEO = True

# The dendrogram walk and the leaf-collection helper both recurse to the depth of a
# Ward tree over ~2.7k clips, which is not bounded by log2(n).
sys.setrecursionlimit(100000)


def load_model_and_processor(model=None, device="cuda"):
    model, processor, _ = load_topo_omni(model=model, device=device)
    return model, processor


def ensure_device_dtype(batch: dict, device: torch.device):
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            continue
        v = v.to(device)
        if device.type == "cuda" and v.is_floating_point():
            v = v.to(torch.bfloat16)
        batch[k] = v
    return batch


def cohen_d(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    mean1, mean2 = x1.mean(axis=0), x2.mean(axis=0)
    std1, std2 = x1.std(axis=0, ddof=1), x2.std(axis=0, ddof=1)
    n1, n2 = x1.shape[0], x2.shape[0]
    pooled = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) /
                     max(n1 + n2 - 2, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (mean1 - mean2) / pooled
        d[~np.isfinite(d)] = 0.0
    return d


def _bh_fallback(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values if SciPy FDR not available."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    q = np.empty_like(p)
    cummin = 1.0
    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = n - rank_from_end + 1
        val = p[idx] * n / rank
        cummin = min(cummin, val)
        q[idx] = cummin
    return np.minimum(q, 1.0)


def fdr_qvalues(p: np.ndarray) -> np.ndarray:
    try:
        return stats.false_discovery_control(p)
    except Exception:
        return _bh_fallback(p)


def run_stats(target: np.ndarray, baseline: np.ndarray):
    tvals, pvals = stats.ttest_ind(target, baseline, axis=0, equal_var=False,
                                   alternative='greater')
    dvals = cohen_d(target, baseline)
    return dict(t=tvals, p=pvals, d=dvals)


@torch.no_grad()
def extract_features_for_video(
    model,
    processor,
    video_paths: List[str],
    output_dir: str,
    device: str = "cuda",
    use_audio_in_video: bool = USE_AUDIO_IN_VIDEO,
    prompt: str = None,
):
    """Mean unified cortical sheet over a clip's time steps, saved as ``<clip>.npy``.

    Returns the ``(n_row, n_col)`` sheet. Video is fed with its audio and no text.
    """
    model.eval()
    dev = torch.device(device)
    processor.tokenizer.padding_side = "right"

    video_id = os.path.basename(video_paths[0])

    if prompt is not None:
        chats = [[
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": [{"type": "video", "video": video}]},
        ] for video in video_paths]
    else:
        chats = [[{"role": "user", "content": [{"type": "video", "video": video}]}]
                 for video in video_paths]

    text = processor.apply_chat_template(chats, add_generation_prompt=False, tokenize=False)

    audios, images, videos = process_mm_info(chats, use_audio_in_video=use_audio_in_video)

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )

    inputs = ensure_device_dtype(inputs, dev)

    out = model(**inputs, return_dict=True)

    # (T, n_row, n_col) already detached to CPU by the model's forward.
    unified_sheet = out.unified_sheet
    unified_sheet = unified_sheet.mean(dim=0)
    unified_sheet = unified_sheet.float().numpy()
    # Write-then-rename: several shard workers share this cache directory, and a
    # half-written .npy is indistinguishable from a good one on the next pass.
    dest = os.path.join(output_dir, f"{video_id}.npy")
    tmp = f"{dest}.{os.getpid()}.tmp.npy"
    np.save(tmp, unified_sheet)
    os.replace(tmp, dest)
    return unified_sheet


def make_cluster_score_fn(min_size: int = 10, max_size: int = 500):
    """Median Welch t of a candidate cluster against everything outside it.

    Clusters outside ``[min_size, max_size]`` score ``-inf``: below the floor the
    t-test is noise, above the ceiling a "cluster" is just a large chunk of the
    stimulus set. The bounds are what the cached tree filename records.
    """
    def cluster_score_fn(target: np.ndarray, baseline: np.ndarray) -> float:
        if target.shape[0] == 0 or baseline.shape[0] == 0:
            return -np.inf
        if target.shape[0] < min_size or target.shape[0] > max_size:
            return -np.inf
        # Only the t map is scored, so call ttest_ind rather than run_stats -- the
        # Cohen's d it also computes is another two passes over a multi-GB baseline.
        tvals, _ = stats.ttest_ind(target, baseline, axis=0, equal_var=False,
                                   alternative='greater')
        return float(np.median(tvals))

    cluster_score_fn.min_size = min_size
    cluster_score_fn.max_size = max_size
    return cluster_score_fn


def clip_seconds(path):
    """Duration of a clip in seconds, or ``inf`` if ffprobe cannot say."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception:
        return float('inf')


def load_or_extract_sheets(model, processor, video_ids, output_dir, spacetop_dir=None,
                           device="cuda", shard=0, num_shards=1, max_clip_seconds=None):
    """Cortical sheet per clip, extracting only what is not already cached.

    Returns ``(kept_idx, kept_ids, sheets)`` with ``sheets`` an ``(N, n_row*n_col)``
    float32 matrix in the order of ``kept_ids``, and ``kept_idx`` their positions in
    the input list so the caller can subset the clip embeddings the same way. Clips
    whose sheet contains NaN are dropped (a few clips have no decodable audio).
    """
    cortical_sheets_dir = os.path.join(output_dir, 'cortical_sheets')
    os.makedirs(cortical_sheets_dir, exist_ok=True)

    # Extraction is decode-bound, not GPU-bound, so it parallelises across processes
    # sharing one GPU. Workers interleave the clip list and skip what is cached, so
    # they finish together and a resumed shard costs one stat() per done clip.
    todo = list(enumerate(video_ids))[shard::num_shards]

    kept_idx, kept_ids, kept_sheets = [], [], []
    for pos, vid in tqdm(todo, desc=f"Cortical sheets [{shard + 1}/{num_shards}]"):
        video_id = os.path.basename(vid)
        cache = os.path.join(cortical_sheets_dir, f"{video_id}.npy")
        if os.path.exists(cache):
            sheet = np.load(cache)
        else:
            clip_path = resolve_clip(vid, spacetop_dir)
            if not os.path.exists(clip_path):
                print(f"  ! missing clip, skipping: {clip_path}")
                continue
            # Clips are cumulative prefixes of their movie, so they run from 2s to
            # several minutes and the longest ones need more GPU memory than exists.
            # Capping lets a small-GPU worker spend its time on clips it can finish
            # and leaves the rest for a bigger one -- the sheet cache merges the two.
            if max_clip_seconds is not None:
                secs = clip_seconds(clip_path)
                if secs > max_clip_seconds:
                    print(f"  - over {max_clip_seconds:.0f}s ({secs:.0f}s), leaving for a "
                          f"larger GPU: {os.path.basename(clip_path)}", flush=True)
                    continue
            try:
                sheet = extract_features_for_video(
                    model, processor, [clip_path], cortical_sheets_dir, device=device)
            except Exception as exc:
                # Clips are cumulative prefixes of their movie (2s, 4s, 6s, ...), so the
                # tail of a long movie is a minutes-long clip that can exhaust the GPU.
                # Drop the allocator's cache before moving on or the failure compounds.
                print(f"  ! failed on {clip_path}: "
                      f"{str(exc).splitlines()[0] if str(exc) else type(exc).__name__}",
                      flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        if np.isnan(sheet).any():
            continue
        kept_idx.append(pos)
        kept_ids.append(vid)
        if num_shards == 1:
            kept_sheets.append(np.asarray(sheet, dtype=np.float32).reshape(-1))

    sheets = np.stack(kept_sheets, axis=0) if kept_sheets else np.empty((0, 0), np.float32)
    return kept_idx, kept_ids, sheets


def cluster_with_early_stopping(embeddings, video_ids, sheets, score_fn):
    """Top-down walk of the Ward tree, splitting only while the score improves.

    Args:
        embeddings: ``(N, D)`` clip embeddings, aligned with ``video_ids``.
        video_ids:  clip ids, aligned with the rows of ``sheets``.
        sheets:     ``(N, U)`` cortical sheets, one row per clip.
        score_fn:   ``(target, baseline) -> float`` over sheet matrices.

    Returns a list of clusters, each a list of clip ids.
    """
    n = len(video_ids)
    Z = linkage(embeddings, method='ward', metric='euclidean')

    # Leaves of every node, bottom-up. Z's rows are ordered so both children of a
    # merge are already built; this replaces a recursive walk repeated per node.
    node_leaves = [[i] for i in range(n)]
    for row in range(Z.shape[0]):
        left, right = int(Z[row, 0]), int(Z[row, 1])
        node_leaves.append(node_leaves[left] + node_leaves[right])

    min_size = getattr(score_fn, "min_size", 0)
    max_size = getattr(score_fn, "max_size", n)

    def score(leaves):
        # Skip materialising the baseline for sizes the score rejects outright --
        # near the root that is every node, and the baseline copy is GB-scale.
        if len(leaves) < min_size or len(leaves) > max_size:
            return -np.inf
        mask = np.ones(n, dtype=bool)
        mask[leaves] = False
        return score_fn(sheets[leaves], sheets[mask])

    def split(node_id):
        leaves = node_leaves[node_id]
        parent_score = score(leaves)

        if node_id < n:          # leaf: cannot split further
            return [[video_ids[i] for i in leaves]]

        row = int(node_id - n)
        left_node, right_node = int(Z[row, 0]), int(Z[row, 1])
        left_leaves, right_leaves = node_leaves[left_node], node_leaves[right_node]

        left_score = score(left_leaves)
        right_score = score(right_leaves)

        print(f"Node {node_id}: parent_score={parent_score:.7f}, "
              f"left_score={left_score:.7f}, right_score={right_score:.7f}", flush=True)

        if left_score < parent_score and right_score < parent_score:
            return [[video_ids[i] for i in leaves]]
        elif left_score < parent_score:
            return [[video_ids[i] for i in left_leaves]] + split(right_node)
        elif right_score < parent_score:
            return split(left_node) + [[video_ids[i] for i in right_leaves]]

        return split(left_node) + split(right_node)

    return split(2 * n - 2)


def main():
    parser = argparse.ArgumentParser(
        description='Discover clusters of SpaceTop clips by walking a Ward dendrogram '
                    'with cortical-sheet-guided early stopping.')
    add_run_args(parser)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint dir to load. Default: $TOPO_OMNI_MODEL.')
    parser.add_argument('--vectors_path', type=str, default=None,
                        help='clip embeddings .npy. Default: <output_dir>/video_clip_vectors.npy')
    parser.add_argument('--manifest_path', type=str, default=None,
                        help='clips manifest .json. Default: <output_dir>/clips_manifest.json')
    parser.add_argument('--min_cluster', type=int, default=10,
                        help='clusters smaller than this score -inf.')
    parser.add_argument('--max_cluster', type=int, default=500,
                        help='clusters larger than this score -inf.')
    parser.add_argument('--linkage_name', type=str, default=None,
                        help='cached tree filename. Default: linkage_tree_tvals_min=<>_max=<>.pkl')
    parser.add_argument('--max_clip_seconds', type=float, default=None,
                        help='Skip clips longer than this. Peak GPU memory scales with '
                             'clip length; ~58s is the limit of an 80GiB A100 on the '
                             'tiled32 sheet. Skipped clips stay uncached for a bigger GPU.')
    parser.add_argument('--shard', type=int, default=0,
                        help='Which slice of the clip list this worker extracts. '
                             'Only meaningful with --extract_only.')
    parser.add_argument('--num_shards', type=int, default=1,
                        help='How many extraction workers share the sheet cache.')
    parser.add_argument('--extract_only', action='store_true',
                        help='extract the cortical sheets and stop (they are the slow, '
                             'GPU-bound part; the walk itself is CPU-only).')
    args = resolve_run_args(parser.parse_args())

    out = args.output_dir
    vectors_path = args.vectors_path or os.path.join(out, 'video_clip_vectors.npy')
    manifest_path = args.manifest_path or os.path.join(out, 'clips_manifest.json')
    linkage_name = args.linkage_name or \
        f'linkage_tree_tvals_min={args.min_cluster}_max={args.max_cluster}.pkl'
    linkage_path = os.path.join(out, linkage_name)

    for path, what in [(vectors_path, 'Embeddings'), (manifest_path, 'Manifest')]:
        if not os.path.exists(path):
            print(f"Error: {what} file not found: {path}")
            sys.exit(1)

    print(f"Loading embeddings from {vectors_path}...")
    embeddings = np.load(vectors_path)
    print(f"Loaded embeddings shape: {embeddings.shape}")

    print(f"Loading clips manifest from {manifest_path}...")
    clips_manifest = read_json(manifest_path)
    print(f"Loaded {len(clips_manifest)} clips")

    if len(embeddings) != len(clips_manifest):
        print(f"Warning: Mismatch between embeddings ({len(embeddings)}) and "
              f"clips ({len(clips_manifest)})")
        min_len = min(len(embeddings), len(clips_manifest))
        print(f"Using min length: {min_len}")
        embeddings = embeddings[:min_len]
        clips_manifest = clips_manifest[:min_len]

    if os.path.exists(linkage_path) and not args.extract_only:
        print(f"Found cached tree at {linkage_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model_and_processor(model=args.checkpoint, device=device)

    if args.num_shards > 1 and not args.extract_only:
        parser.error("--num_shards > 1 only makes sense with --extract_only; the "
                     "dendrogram walk needs every clip's sheet at once.")

    keep, video_ids, sheets = load_or_extract_sheets(
        model, processor, clips_manifest, out, spacetop_dir=args.spacetop_dir,
        device=device, shard=args.shard, num_shards=args.num_shards,
        max_clip_seconds=args.max_clip_seconds)

    if args.extract_only:
        print(f"--extract_only: {len(video_ids)} sheets cached by shard "
              f"{args.shard}/{args.num_shards}.")
        return

    embeddings = embeddings[keep]
    print(f"Using {len(video_ids)} videos for clustering "
          f"(sheets {sheets.shape}) after filtering out NaN/missing")

    # The walk is pure numpy over `sheets`; free the GPU copy of the model first.
    del model, processor
    if device == "cuda":
        torch.cuda.empty_cache()

    # The walk drops clips with a missing or NaN sheet, so its Ward tree is built over
    # a subset of the manifest. Record that subset: plot_dendrogram rebuilds the same
    # linkage matrix, and rebuilding it from the full manifest gives a different tree
    # whose leaf indices no longer line up with the clusters.
    used_path = os.path.join(out, 'clips_used.json')
    write_json(video_ids, used_path)
    print(f"Wrote {used_path} ({len(video_ids)} clips)")

    score_fn = make_cluster_score_fn(args.min_cluster, args.max_cluster)
    tree = cluster_with_early_stopping(embeddings, video_ids, sheets, score_fn)
    print(f"Number of clusters: {len(tree)}")
    write_pickle(tree, linkage_path)
    print(f"Wrote {linkage_path}")


if __name__ == "__main__":
    main()
