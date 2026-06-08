from typing import List, Tuple, Dict, Any

import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import scipy.stats as stats

from scipy.cluster.hierarchy import linkage, fcluster

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig

from qwen_omni_utils import process_mm_info

import sys
sys.path.append("../src")
from models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

from dotenv import load_dotenv
load_dotenv()

CKPT_DIR = os.getenv("CKPT_DIR")
STIMULI_DIR = os.getenv("STIMULI_DIR")

USE_AUDIO_IN_VIDEO = True

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pkl.dump(data, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pkl.load(f)

def load_model_and_processor(run_dir, device="cuda"):
    processor = Qwen2_5OmniProcessor.from_pretrained(run_dir)
    model_config = Qwen2_5OmniThinkerConfig.from_pretrained(run_dir)

    model_config.audio_config.is_training = False
    model_config.vision_config.is_training = False
    model_config.text_config.is_training = False
    model_config.apply_spatial_loss = True

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        run_dir,
        config=model_config,
        device_map=torch.device(device),
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    )

    model.eval()
    model.bfloat16()
    return model, processor

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

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
    """Benjamini–Hochberg q-values if SciPy FDR not available."""
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
    tvals, pvals = stats.ttest_ind(target, baseline, axis=0, equal_var=False, alternative='greater')
    # qvals = fdr_qvalues(pvals.flatten())
    dvals = cohen_d(target, baseline)
    return dict(
        t=tvals, 
        p=pvals, 
        d=dvals, 
        # q=qvals, 
    )

@torch.no_grad()
def extract_features_for_video(
    model,
    processor,
    video_paths: List[str],
    output_dir: str,
    batch_size: int = 1,
    device: str = "cuda",
    use_audio_in_video: bool = True,
    prompt: str|None = None,
):
    """
    Returns: feats_lm (N, D_lm) or None, feats_vis (N, D_vis) or None.
    Feeds audio with NO accompanying text. Uses the chat template with <audio> only.
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
        chats = [[{"role": "user", "content": [{"type": "video", "video": video}]}] for video in video_paths]

    text = processor.apply_chat_template(
        chats, 
        add_generation_prompt=False, 
        tokenize=False,
    )

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

    unified_sheet = out.unified_sheet

    unified_sheet = unified_sheet.mean(dim=0)
    unified_sheet = unified_sheet.float().numpy()
    np.save(f"{output_dir}/{video_id}.npy", unified_sheet)
    return unified_sheet

def extract_features_for_video_parallel(model, processor, video_paths, batch_size=1):
    # Simple parallelization using multiprocessing Pool
    with Pool(processes=cpu_count()) as pool:
        func = partial(extract_features_for_video, model, processor, batch_size=batch_size)
        results = pool.map(func, [video_paths[i:i + batch_size] for i in range(0, len(video_paths), batch_size)])
    return np.concatenate(results, axis=0)

def cluster_score_fn(target_sheets, baseline_sheets):
    if len(target_sheets) == 0 or len(baseline_sheets) == 0:
        return -np.inf

    if len(target_sheets) < 10 or len(target_sheets) > 500:
        return -np.inf
    
    target_sheets = np.stack(target_sheets).reshape(len(target_sheets), -1)
    baseline_sheets = np.stack(baseline_sheets).reshape(len(baseline_sheets), -1)
    scores = run_stats(target_sheets, baseline_sheets)['t']
    score = np.median(scores)  # Use median Cohen's d as the cluster score
    return score

def cluster_with_early_stopping(model, processor, embeddings, video_ids, score_fn, output_dir):
    """
    Agglomerative clustering with early stopping based on a score function.
    
    Args:
        model: The model for feature extraction
        processor: The processor for handling inputs
        embeddings: (N, D) array of embeddings
        video_ids: list of video IDs corresponding to each embedding
        score_fn: function that takes a list of video IDs and returns a score
    
    Returns:
        list of clusters (each cluster is a list of video IDs)
    """
    cortical_sheets_dir = os.path.join(output_dir, 'cortical_sheets')
    # cortical_sheets_dir = cortical_sheets_dir.replace("clustering_v3", "clustering_v2")
    if not os.path.exists(cortical_sheets_dir):
        os.makedirs(cortical_sheets_dir)

    cortical_sheets = {}
    filtered_video_ids = []
    filtered_embeddings = []
    for idx, vid in tqdm(enumerate(video_ids), total=len(video_ids), desc="Extracting features for videos"):
        video_id = os.path.basename(vid)
        if not os.path.exists(os.path.join(cortical_sheets_dir, f"{video_id}.npy")):
            print(f"> Extracting features for {vid}...")
            cortical_sheets_npy = extract_features_for_video(model, processor, [vid], cortical_sheets_dir, batch_size=1)
        else:
            cortical_sheets_npy = np.load(os.path.join(cortical_sheets_dir, f"{video_id}.npy"))
        
        if not np.isnan(cortical_sheets_npy).any():
            cortical_sheets[vid] = cortical_sheets_npy
            filtered_video_ids.append(vid)
            filtered_embeddings.append(embeddings[idx])

    
    video_ids = filtered_video_ids
    embeddings = np.array(filtered_embeddings)
    print(f"Using {len(video_ids)} videos for clustering after filtering out NaN features")
          
    # Step 1: Build the full linkage matrix
    # Z[i] = [idx1, idx2, distance, cluster_size]
    Z = linkage(embeddings, method='ward', metric='euclidean')
    
    n = len(video_ids)

    def sample_baseline(exclude_set, k=200):
        pool = [vid for vid in video_ids if vid not in exclude_set]
        return [cortical_sheets[vid] for vid in pool]
        k = min(k, len(pool))
        idx = np.random.choice(len(pool), size=k, replace=False)
        return [cortical_sheets[pool[i]] for i in idx]
    
    # Step 2: Walk the dendrogram top-down and decide whether to split
    # Each node in the linkage matrix with index (n + i) merges Z[i,0] and Z[i,1]
    
    # Build a mapping: node_id -> children
    # Leaf nodes are 0..n-1, internal nodes are n..2n-2
    def get_leaves(node_id):
        """Recursively get all leaf indices under a node."""
        if node_id < n:
            return [node_id]
        row = int(node_id - n)
        left = int(Z[row, 0])
        right = int(Z[row, 1])
        return get_leaves(left) + get_leaves(right)
    
    # Step 3: Top-down recursive splitting with early stopping
    def split(node_id):
        """
        Returns a list of clusters (each cluster = list of video IDs).
        Splits only if both children score higher than the parent.
        """
        leaves = get_leaves(node_id)
        parent_ids = [video_ids[i] for i in leaves]
        parent_score = score_fn([cortical_sheets[vid] for vid in parent_ids], sample_baseline(parent_ids))
        
        # Leaf node — can't split further
        if node_id < n:
            return [parent_ids]
        
        row = int(node_id - n)
        left_node = int(Z[row, 0])
        right_node = int(Z[row, 1])
        
        left_leaves = get_leaves(left_node)
        right_leaves = get_leaves(right_node)
        
        left_ids = [video_ids[i] for i in left_leaves]
        right_ids = [video_ids[i] for i in right_leaves]
        
        left_score = score_fn([cortical_sheets[vid] for vid in left_ids], sample_baseline(left_ids))
        right_score = score_fn([cortical_sheets[vid] for vid in right_ids], sample_baseline(right_ids))
        
        # Stop splitting if either child scores lower
        print(f"Node {node_id}: parent_score={parent_score:.7f}, left_score={left_score:.7f}, right_score={right_score:.7f}")
        if left_score < parent_score and right_score < parent_score:
            return [parent_ids]
        elif left_score < parent_score:
            return [left_ids] + split(right_node)
        elif right_score < parent_score:
            return split(left_node) + [right_ids]
        
        # Both children are at least as good — recurse
        return split(left_node) + split(right_node)
    
    # Root node is the last merge: index = 2n - 2
    root = 2 * n - 2
    return split(root)

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    from scipy.cluster.hierarchy import linkage, fcluster
    import matplotlib.pyplot as plt
    import os
    import sys
    import json
    import argparse
    from pathlib import Path
    from multiprocessing import Pool, cpu_count
    from functools import partial

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run agglomerative clustering on video clip embeddings')
    parser.add_argument('--vectors_path', type=str,
                    default='task-alignvideo/clustering_v2/video_clip_vectors.npy',
                    help='Path to video clip embeddings .npy file')
    parser.add_argument('--manifest_path', type=str,
                    default='task-alignvideo/clustering_v2/clips_manifest.json',
                    help='Path to clips manifest JSON file')
    parser.add_argument('--output_dir', type=str,
                    default='task-alignvideo/clustering_v2',
                    help='Output directory for clustering results')
    args = parser.parse_args()

    vectors_path = args.vectors_path
    manifest_path = args.manifest_path
    main_output_dir = args.output_dir

    # Validate input files exist
    if not os.path.exists(vectors_path):
        print(f"Error: Embeddings file not found: {vectors_path}")
        sys.exit(1)

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found: {manifest_path}")
        sys.exit(1)

    # Load embeddings
    print(f"Loading embeddings from {vectors_path}...")
    embeddings = np.load(vectors_path)
    print(f"Loaded embeddings shape: {embeddings.shape}")

    # Load clips manifest
    print(f"Loading clips manifest from {manifest_path}...")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        clips_manifest = json.load(f)
    print(f"Loaded {len(clips_manifest)} clips")

    # Verify lengths match
    if len(embeddings) != len(clips_manifest):
        print(f"Warning: Mismatch between embeddings ({len(embeddings)}) and clips ({len(clips_manifest)})")
        print(f"Using min length: {min(len(embeddings), len(clips_manifest))}")
        min_len = min(len(embeddings), len(clips_manifest))
        embeddings = embeddings[:min_len]
        clips_manifest = clips_manifest[:min_len]
    
    file_names = clips_manifest

    linkage_path = os.path.join(main_output_dir, 'linkage_tree_tvals_min=10_max=500.pkl')
    if os.path.exists(linkage_path):
        print(f"Found cached tree at {linkage_path}")
        tree = load_pickle(linkage_path)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = os.path.join(CKPT_DIR, "qwen2_5_3b_spatial_task_final_7")
        model, processor = load_model_and_processor(model_path, device=device)
        tree = cluster_with_early_stopping(model, processor, embeddings, file_names, cluster_score_fn, main_output_dir)
        print(f"Number of clusters: {len(tree)}")
        # assert sum(len(c) for c in tree) == len(file_names)
        save_pickle(tree, linkage_path)
