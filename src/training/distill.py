import os
import json
import torch
import numpy as np

from tqdm import tqdm

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig
from qwen_omni_utils import process_mm_info

from transformers import Qwen2_5OmniThinkerForConditionalGeneration

from dotenv import load_dotenv
load_dotenv()

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
DATA_DIR = os.getenv("DATA_DIR")

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def generate(model, processor, prompt, filename):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": f"{DATA_DIR}/{filename}.mp4"},
                {"type": "text", "text": prompt}
            ],
        }
    ]

    # set use audio in video
    USE_AUDIO_IN_VIDEO = True

    # Preparation for inference
    text = processor.apply_chat_template(
        conversation, 
        add_generation_prompt=True, 
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

    with torch.inference_mode():
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            max_new_tokens=128, 
            min_new_tokens=16,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = text_ids[:, input_len:]

    text_out = processor.batch_decode(
        new_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]

    return text_out.split("<|im_end|>")[0].strip()

if __name__ == "__main__":

    np.random.seed(42)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = Qwen2_5OmniThinkerConfig.from_pretrained(MODEL_ID)

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        dtype='auto', 
        device_map="auto",
        config=model_config,
    )

    model.eval()
    model.bfloat16()

    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)

    dataset = read_json(f"data/koala_final.json")

    savepath = "data/koala_final_distilled.json"

    new_dataset = read_json(savepath) if os.path.exists(savepath) else []
    processed = set([row["id"] for row in new_dataset])

    for row in tqdm(dataset):

        video_id = row["id"]
        if video_id in processed:
            print(f"> Skipping already processed video {video_id}")
            continue

        # random_question = np.random.choice(questions)
        question = row["prompt"]

        print("Question:", question)
        print("Video Filename:", video_id)

        try:
            output = generate(model, processor, question, video_id)
            print("Output:", output)

            row["prompt"] = question
            row["model_response"] = output

            new_dataset.append(row)

            write_json(savepath, new_dataset)
        except:
            print(f"> Error processing video {video_id}, skipping.")
            continue 

    



    
 