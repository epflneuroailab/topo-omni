import os
import json
import torch

from transformers import DataCollatorForLanguageModeling
from typing import Union, List, Any, Dict, Optional
from transformers import Qwen2_5OmniProcessor

from datasets import load_dataset
from qwen_omni_utils import process_mm_info

from dotenv import load_dotenv
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")
#: Root for image manifests that carry a bare `id` instead of an absolute `path`.
#: Videos resolve as {DATA_DIR}/{id}.mp4; images as {IMAGE_DIR}/{id}.jpg unless the
#: manifest row gives an explicit `path`.
IMAGE_DIR = os.getenv("IMAGE_DIR")

def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
    
def read_json(path):
    with open(path, 'r') as fin:
        data = json.load(fin)
    return data

def clean_conversation(conversation):
    cleaned = []
    for m in conversation:
        new_content = []
        for c in m["content"]:
            # enforce fixed schema
            item = {
                "type": c.get("type"),
                "text": c.get("text", None),
                "audio": c.get("audio", None),
                "video": c.get("video", None),
                "image": c.get("image", None),
                "resized_height": c.get("resized_height", None),
                "resized_width": c.get("resized_width", None),
            }

            item = {k: v for k, v in item.items() if v is not None}
            # keep only if it contains something meaningful
            if not (item.get("text") is None and item.get("audio") is None
                    and item.get("video") is None and item.get("image") is None):
                new_content.append(item)

        cleaned.append({"role": m["role"], "content": new_content})
    return cleaned


def build_loss_mask(input_ids: torch.Tensor, pattern_ids):
    """
    input_ids : LongTensor of shape [batch_size, seq_len]
    pattern_ids : list or 1D torch.Tensor, e.g. [151644, 77091]

    Returns:
        loss_mask : BoolTensor [batch_size, seq_len]
                    True  = include in loss
                    False = set label = -100
    """
    assert input_ids.dim() == 2, "input_ids must be [batch_size, seq_len]"

    bsz, seq_len = input_ids.size()
    pattern = torch.as_tensor(pattern_ids, device=input_ids.device, dtype=input_ids.dtype)
    L = pattern.numel()

    # If pattern longer than sequence → mask all as False
    if seq_len < L:
        return torch.zeros_like(input_ids, dtype=torch.bool)

    # Create sliding windows across seq_len
    # shape → [bsz, seq_len-L+1, L]
    windows = input_ids.unfold(dimension=1, size=L, step=1)

    # Compare with pattern across last dim
    # shape → [bsz, seq_len-L+1]
    match_matrix = (windows == pattern).all(dim=-1)

    # loss mask starts empty
    loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    # For each sample in batch, find first match and mask tokens AFTER pattern
    for i in range(bsz):
        match_positions = match_matrix[i].nonzero(as_tuple=True)[0]
        if len(match_positions) > 0:
            start = match_positions[0].item() + L  # first token AFTER pattern
            loss_mask[i, start:] = True  # enable loss only after pattern

    return loss_mask
    
class DataCollatorForQwenOmni(DataCollatorForLanguageModeling):
    
    def __init__(
        self,
        processor,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, tokenizer=processor.tokenizer, **kwargs)
        self.processor = processor
        self.ignore_index = ignore_index

    def preprocess_fn(self, batch):
        conversations = [clean_conversation(json.loads(example["conversation_json"]))
                         for example in batch]
        texts = [example["text"] for example in batch]

        # The thinker forward indexes `audio_time_steps_all` by BATCH POSITION but only
        # appends to it for items that actually carry audio, and it calls .item() on an
        # audio-start-token search that is empty for an image. A batch holding both dies
        # with "a Tensor with 0 elements cannot be converted to Scalar", ~30 s into the
        # step. Fail here instead, where the cause is legible.
        mods = {example.get("modality", "video") for example in batch}
        if len(mods) > 1:
            raise ValueError(
                f"batch mixes modalities {sorted(mods)}; the thinker forward cannot pool "
                f"audio-bearing and audio-free items together. Use ModalityBatchSampler "
                f"(src/training/batching.py), which yields modality-homogeneous batches."
            )
        use_audio = mods == {"video"}

        audios, images, videos = process_mm_info(conversations, use_audio_in_video=use_audio)

        outputs = self.processor(
            text=texts,
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=use_audio, 
        )
    
        input_ids = outputs["input_ids"]

        loss_mask = build_loss_mask(input_ids, [151644, 77091])
        labels = input_ids.clone()
        labels[~loss_mask] = -100   # ignore everywhere except after assistant pattern

        outputs["labels"] = labels

        return outputs
    

    def torch_call(self, batch: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        return self.preprocess_fn(batch)
    

def load_and_format_dataset(data_path, processor: Qwen2_5OmniProcessor, debug: bool = False,
                            image_side: int = 224):
    """
    Expect a JSON/JSONL with fields:
        - type: "video" | "image"
        - id (+ optional absolute `path` for images)
        - prompt, model_response

    `data_path` may be a single path or a LIST of paths, which `datasets` concatenates.
    That is how an image manifest is trained alongside the video one; the two are kept
    in separate batches downstream by ModalityBatchSampler, never mixed within a batch
    (see src/training/batching.py for why mixing crashes).

    `image_side` resizes every image to exactly image_side x image_side. This is NOT
    cosmetic. In the thinker forward's no-audio path `text_time_steps` is initialised
    once before the per-item loop and then ratcheted down inside it, and is only reset
    per item on the audio branch -- so if items have different visual token counts, the
    pooling collapses. Measured on 16 DTD images at native size: T = 2 instead of 16,
    silently discarding ~87% of the spatial loss's sample axis.

    224 is the default because it is what the orientation eval's sine gratings already
    produce (256 patches -> 64 tokens), so training and eval present the vision encoder
    with identically shaped inputs.
    """

    # datasets handles both json and jsonl the same way.
    #
    # Manifests are loaded ONE AT A TIME and narrowed to the columns `to_messages` reads
    # before being concatenated. Passing several files to `data_files` instead makes
    # `datasets` demand one shared schema across them, and the video manifest carries 13
    # fields (url, timestamp, the four quality scores, duration_ms, ...) that an image
    # manifest has no analogue for -- which fails with DatasetGenerationCastError rather
    # than filling nulls.
    KEEP = ("type", "id", "prompt", "model_response", "path")
    paths = list(data_path) if isinstance(data_path, (list, tuple)) else [data_path]
    parts = []
    for path in paths:
        d = load_dataset("json", data_files=path, split="train")
        missing = [c for c in ("type", "id", "prompt", "model_response")
                   if c not in d.column_names]
        if missing:
            raise ValueError(f"{path} is missing required manifest fields {missing}")
        if "path" not in d.column_names:
            d = d.add_column("path", [None] * len(d))
        parts.append(d.select_columns(list(KEEP)))
    if len(parts) == 1:
        ds = parts[0]
    else:
        from datasets import concatenate_datasets
        ds = concatenate_datasets(parts)
        print(f"> concatenated {len(parts)} manifests -> {len(ds):,} rows")

    if debug:
        ds = ds.shuffle(seed=42).select(range(100))

    def clean_conversation(conversation):
        cleaned = []
        for m in conversation:
            new_content = []
            for c in m["content"]:
                # enforce fixed schema
                item = {
                    "type": c.get("type"),
                    "text": c.get("text", None),
                    "audio": c.get("audio", None),
                    "video": c.get("video", None),
                    "image": c.get("image", None),
                    # exact resize; see IMAGE_SIDE in load_and_format_dataset
                    "resized_height": c.get("resized_height", None),
                    "resized_width": c.get("resized_width", None),
                }

                item = {k: v for k, v in item.items() if v is not None}
                # keep only if it contains something meaningful
                if not (item.get("text") is None and item.get("audio") is None
                        and item.get("video") is None and item.get("image") is None):
                    new_content.append(item)

            cleaned.append({"role": m["role"], "content": new_content})
        return cleaned

    
    def to_messages(example):
        if example["type"] == "video":
            stimuli_path = f"{DATA_DIR}/{example['id']}.mp4"
            stimuli_type = "video"
        elif example["type"] == "image":
            # An image manifest may carry an absolute `path` (what build_image_manifest
            # writes) or just an `id` to resolve under $IMAGE_DIR, matching the video
            # convention.
            stimuli_path = example.get("path") or f"{IMAGE_DIR}/{example['id']}.jpg"
            stimuli_type = "image"
        else:
            raise ValueError(f"Unknown type: {example['type']}")

        prompt = example["prompt"]

        mm_item = {
            "type": stimuli_type,
            "audio": stimuli_path if stimuli_type == "audio" else None,
            "video": stimuli_path if stimuli_type == "video" else None,
            "image": stimuli_path if stimuli_type == "image" else None,
        }
        if stimuli_type == "image":
            # EVERY IMAGE MUST YIELD THE SAME VISUAL TOKEN COUNT, or T collapses --
            # see the note on `image_side` in this function's signature. An explicit
            # resized_height/width forces an exact grid; the processor's own
            # min_pixels/max_pixels cannot, because smart_resize preserves aspect ratio
            # and rounds each side to a multiple of 28 (measured: pinning the pixel
            # budget still left 234..272 tokens across 8 DTD images, while an explicit
            # 224x224 gives exactly 256 patches / 64 tokens for all of them).
            mm_item["resized_height"] = image_side
            mm_item["resized_width"] = image_side

        text_item = {
            "type": "text",
            "text": prompt,
        }

        messages = [
            {"role": "user", "content": [mm_item, text_item]},
            {"role": "assistant", "content": [{"type": "text", "text": example["model_response"]}]}
        ]
        # Serialised, NOT returned as a nested struct. `datasets` infers an arrow schema
        # per map shard, and the content structs differ between shards: a video row has
        # image=None while an image row has image="/path", so alignment fails with
        # "The features can't be aligned because the key messages ... has unexpected
        # type". A JSON string has one schema everywhere.
        #
        # `modality` is what ModalityBatchSampler groups on, so it must survive both
        # .map calls -- returned here and re-emitted by `tokenize`.
        return {"messages_json": json.dumps(messages), "modality": stimuli_type}

    
    def tokenize(example):
    
        conversation = clean_conversation(json.loads(example["messages_json"]))
        # An image has no audio track, so the audio placeholders use_audio_in_video
        # inserts would never be filled. Keep it True for video so existing runs
        # tokenise byte-identically.
        use_audio = example["modality"] == "video"

        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=False,  
            tokenize=False,
            use_audio_in_video=use_audio, 
        )

        outputs = processor(
            text=text, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=use_audio, 
        )

        # Serialised for the same reason as messages_json above; the collator parses it.
        outputs["conversation_json"] = json.dumps(conversation)
        outputs["text"] = text
        outputs["modality"] = example["modality"]

        return outputs
    

    ds = ds.map(
        to_messages, 
        num_proc=8,   
        batched=False,    
        remove_columns=ds.column_names,
    )

    ds = ds.map(
        tokenize,
        num_proc=8,
        batched=False,
        remove_columns=ds.column_names,
    )

    return ds
