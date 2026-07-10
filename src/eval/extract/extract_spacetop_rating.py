import os
import json
import torch
import argparse
import numpy as np

from tqdm import tqdm
from glob import glob
from qwen_omni_utils import process_mm_info
from src.core.model_loading import load_topo_omni, MODEL_TITLE

from dotenv import load_dotenv
load_dotenv()

CKPT_DIR = os.getenv("CKPT_DIR")
STIMULI_DIR = os.getenv("STIMULI_DIR")
SAVE_DIR = os.getenv("SAVE_DIR")

USE_AUDIO_IN_VIDEO = False

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def retrieve_cortical_sheet(model, processor, filename):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "In relation to the previous clip, how do you feel?"},
                {"type": "video", "video": filename},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        conversation, 
        add_generation_prompt=False, 
        tokenize=False,
    )

    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(
        text=text, 
        audio=audios, 
        images=images, 
        videos=videos, 
        return_tensors="pt", 
        padding=True, 
        use_audio_in_video=USE_AUDIO_IN_VIDEO, 
    )

    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

    return outputs.unified_sheet.float().cpu().numpy()


if __name__ == "__main__":

    np.random.seed(42)

    model_name = MODEL_TITLE
    DATA_DIR = f"{STIMULI_DIR}/spacetop_rating_v0"
    SAVE_DIR = f"{SAVE_DIR}/{model_name}/spacetop_clusters"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    rating_files = sorted(glob(f"{DATA_DIR}/*.mp4"))

    model, processor, _ = load_topo_omni(device=DEVICE)
    for rating_file in tqdm(rating_files):
        rating_id = os.path.basename(os.path.dirname(rating_file))
        print(f"Processing {rating_id}...")

        filename = os.path.basename(rating_file).replace(".mp4", "")
        save_path = f"{SAVE_DIR}/{rating_id}/{filename}_cortical_sheet.npy"
        if os.path.exists(save_path):
            print(f"Skipping {rating_id}, already exists.")
            continue

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
      
        cortical_sheet = retrieve_cortical_sheet(
            model=model, 
            processor=processor, 
            filename=rating_file
        )

        cortical_sheet = cortical_sheet[-1, ...]  # Get the last time step

        np.save(save_path, cortical_sheet)