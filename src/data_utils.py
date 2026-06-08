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
            }

            item = {k: v for k, v in item.items() if v is not None}
            # keep only if it contains something meaningful
            if not (item.get("text") is None and item.get("audio") is None and item.get("video") is None):
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
        conversations = [clean_conversation(example["conversation"]) for example in batch]
        texts = [example["text"] for example in batch]

        audios, images, videos = process_mm_info(conversations, use_audio_in_video=True)

        outputs = self.processor(
            text=texts,
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=True, 
        )
    
        input_ids = outputs["input_ids"]

        loss_mask = build_loss_mask(input_ids, [151644, 77091])
        labels = input_ids.clone()
        labels[~loss_mask] = -100   # ignore everywhere except after assistant pattern

        outputs["labels"] = labels

        return outputs
    

    def torch_call(self, batch: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        return self.preprocess_fn(batch)
    

def load_and_format_dataset(data_path: str, processor: Qwen2_5OmniProcessor, debug: bool = False):
    """
    Expect a JSON/JSONL with fields:
        - video_path: string path to .mp4
        - text: target text / caption / transcript
    """

    # datasets handles both json and jsonl the same way
    ds = load_dataset("json", data_files=data_path, split="train")

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
                }

                item = {k: v for k, v in item.items() if v is not None}
                # keep only if it contains something meaningful
                if not (item.get("text") is None and item.get("audio") is None and item.get("video") is None):
                    new_content.append(item)

            cleaned.append({"role": m["role"], "content": new_content})
        return cleaned

    
    def to_messages(example):
        if example["type"] == "video":
            stimuli_path = f"{DATA_DIR}/{example['id']}.mp4"
            stimuli_type = "video"
        else:
            raise ValueError(f"Unknown type: {example['type']}")

        prompt = example["prompt"]

        mm_item = {
            "type": stimuli_type,
            "audio": stimuli_path if stimuli_type == "audio" else None,
            "video": stimuli_path if stimuli_type == "video" else None,
        }

        text_item = {
            "type": "text",
            "text": prompt,
        }

        messages = [
            {"role": "user", "content": [mm_item, text_item]},
            {"role": "assistant", "content": [{"type": "text", "text": example["model_response"]}]}
        ]
        return {"messages": messages}

    
    def tokenize(example):
    
        conversation = clean_conversation(example["messages"])

        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=False,  
            tokenize=False,
            use_audio_in_video=True, 
        )

        outputs = processor(
            text=text, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=True, 
        )

        outputs["conversation"] = conversation
        outputs["text"] = text

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
