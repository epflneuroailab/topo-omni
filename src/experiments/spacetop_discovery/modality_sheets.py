"""Per-clip unified sheets from ONE input stream of each SpaceTop clip.

The discovery contrasts are defined on audiovisual clips, so a cluster's map could be
driven by either stream. This re-extracts every clip's mean sheet from a single stream:

  --modality video   the clip's frames only (use_audio_in_video=False)
  --modality audio   the clip's soundtrack only, passed as an audio item with no frames

and writes ``<run_dir>/cortical_sheets_<modality>/<clip>.npy`` in the same format as the
audiovisual cache (``cortical_sheets/``), so ``cluster_maps.py`` / ``cluster_null.py`` can be
pointed at either. Shards interleave the clip list and skip clips already cached, so
several jobs can share one output directory.

    TOPO_SHEET_LAYOUT=tiled32 python -m src.experiments.spacetop_discovery.modality_sheets \\
        --run_dir results/topo-omni-tiled32-4608/spacetop_discovery_disjoint2s_min60 \\
        --checkpoint artifacts/ckpts/topo-omni-tiled32-4608 --modality audio --shard 0 --num_shards 3
"""
import argparse
import json
import os

import numpy as np
import torch
from qwen_omni_utils import process_mm_info
from tqdm import tqdm

from src.experiments.spacetop_discovery.agglomerative_early_stopping import (
    ensure_device_dtype, extract_features_for_video, load_model_and_processor,
)


@torch.no_grad()
def audio_only_sheet(model, processor, clip_path, output_dir, device="cuda"):
    """Mean unified sheet for the clip's soundtrack alone (no frames, no text)."""
    model.eval()
    processor.tokenizer.padding_side = "right"
    chats = [[{"role": "user", "content": [{"type": "audio", "audio": clip_path}]}]]
    text = processor.apply_chat_template(chats, add_generation_prompt=False, tokenize=False)
    audios, images, videos = process_mm_info(chats, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = ensure_device_dtype(inputs, torch.device(device))
    out = model(**inputs, return_dict=True)
    sheet = out.unified_sheet.mean(dim=0).float().numpy()
    dest = os.path.join(output_dir, f"{os.path.basename(clip_path)}.npy")
    tmp = f"{dest}.{os.getpid()}.tmp.npy"
    np.save(tmp, sheet)
    os.replace(tmp, dest)
    return sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run_dir", required=True, help="discovery run dir holding clips_used.json")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--modality", choices=["video", "audio"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "clips_used.json")) as f:
        clips = json.load(f)
    out = os.path.join(args.run_dir, f"cortical_sheets_{args.modality}")
    os.makedirs(out, exist_ok=True)
    mine = [c for i, c in enumerate(clips) if i % args.num_shards == args.shard]
    todo = [c for c in mine if not os.path.exists(os.path.join(out, os.path.basename(c) + ".npy"))]
    print(f"> {args.modality}-only sheets: {len(clips)} clips, shard {args.shard}/{args.num_shards} "
          f"has {len(mine)}, {len(todo)} still to extract -> {out}", flush=True)
    if not todo:
        return

    model, processor = load_model_and_processor(model=args.checkpoint, device=args.device)
    failed = []
    for clip in tqdm(todo, mininterval=30):
        try:
            if args.modality == "video":
                extract_features_for_video(model, processor, [clip], out, device=args.device,
                                           use_audio_in_video=False)
            else:
                audio_only_sheet(model, processor, clip, out, device=args.device)
        except Exception as e:          # keep going; a clip without audio should not kill the shard
            failed.append((os.path.basename(clip), f"{type(e).__name__}: {e}"[:200]))
            if args.device == "cuda":
                torch.cuda.empty_cache()
    print(f"> done: {len(todo) - len(failed)} written, {len(failed)} failed", flush=True)
    for name, err in failed[:20]:
        print(f"   ! {name}: {err}")


if __name__ == "__main__":
    main()
