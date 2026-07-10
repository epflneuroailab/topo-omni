import os
import json
from platform import processor
from pydoc import text
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from transformers import Qwen2_5OmniProcessor, AutoTokenizer
from qwen_omni_utils import process_mm_info

from dotenv import load_dotenv
load_dotenv()

SAVE_DIR = os.getenv("SAVE_DIR")

def read_json(file_path):
    """Read a JSON file and return the data."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def write_json(data, file_path):
    """Write data to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def plot_heatmap(similarity_matrix, save_path, categories, percentage, stimulate=False):
    import seaborn as sns
    import matplotlib.pyplot as plt

    labels = [cat.capitalize() for cat in categories]
    sns.set_theme(style="whitegrid", font_scale=2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap="viridis", cbar=True, fmt=".1f")
    for text in plt.gca().texts:
        text.set_text(text.get_text() + "%")

    plt.title(f"Accuracy Heatmap - Top {percentage}%")
    plt.xticks(ticks=np.arange(0.5, len(similarity_matrix)+0.5), labels=labels)
    plt.yticks(ticks=np.arange(0.5, len(similarity_matrix)+0.5), labels=labels, rotation=0)
    plt.xlabel("Stimuli")

    if stimulate:
        plt.ylabel("Region to Drive")
    else:
        plt.ylabel("Region to Suppress")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    plt.cla()
    plt.close()

def load_model(model_name="Qwen/Qwen2.5-Omni-3B"):
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to("cuda")
    model.eval()
    model.bfloat16()
    return model, processor, tokenizer

def generate_text(model, processor, tokenizer, prompts):
    processor.padding_side = "left"
    conversations = [[{'role': 'user', 'content': [{"type": "text", "text": prompt}]}] for prompt in prompts]
    text = processor.apply_chat_template(conversations, add_generation_prompt=True)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    outputs = [output[output.index("Answer:\nassistant\n") + len("Answer:\nassistant\n"):].strip()for output in outputs]
    return outputs


def compute_similarity(model, percentage):
    results = np.zeros((len(categories), len(categories)))
    for i, stimuli in enumerate(categories):
        no_ablation_path = f"ablation/ablation_stimuli={stimuli}_localizer=no-ablation_v2"
        queries = read_json(f"{SAVE_DIR}/{model_name}/{no_ablation_path}.json")["responses"]
        queries = [item["output"] for item in queries]
        query_embeddings = model.encode(queries, prompt_name="query")
    
        for j, localizer in enumerate(categories):
            
            path = path_template.format(stimuli=stimuli, localizer=localizer, perc=f"{percentage:.2f}")
            data = read_json(f"{SAVE_DIR}/{model_name}/{path}.json")["responses"]

            sentences = [item["output"] for item in data]
            embeddings = model.encode(sentences)
            similarity = model.similarity(query_embeddings, embeddings)
            similarity = np.diag(similarity)
            results[j, i] = np.mean(similarity)
            print(f"Similarity for stimuli={stimuli} localizer={localizer} perc={percentage}: {results[j, i]}")

    return results

if __name__ == "__main__":
    categories = [
        "faces",
        "bodies",
        "scenes",
        "objects",
    ]

    descriptions = [
        "a human face or a close-up of a person",
        "a human body part that is not a face",
        "an indoor or outdoor location",
        "a toy or object",
    ]


    STIMULATE = False
    
    model, processor, tokenizer = load_model("Qwen/Qwen2.5-Omni-3B")
    prompt_template = "Does the following sentence describe {stimuli}?\nAnswer only with Yes or No.\n\nSentence: {sentence}\nAnswer:"
    # prompt_template = "What does the following sentence describe?\nOptions: {options}\n\nSentence: {sentence}\nAnswer:"
    # options = '\n' + "\n".join([f"{k+1}. {desc}" for k, desc in enumerate(descriptions)])

    model_name = "topo-omni"
    path_template = "ablation/ablation_stimuli={stimuli}_localizer=no-ablation_v4"

    results = []
    for i, stimuli in enumerate(categories):
        
        
        path = path_template.format(stimuli=stimuli)
        path = f"{SAVE_DIR}/{model_name}/{path}.json"
        
        data = read_json(path)["responses"]

        sentences = [item["output"] for item in data]

        prompts = [prompt_template.format(stimuli=descriptions[i], sentence=sentence) for sentence in sentences]
        # prompts = [prompt_template.format(options=options, sentence=sentence) for sentence in sentences]
        generated_texts = generate_text(model, processor, tokenizer, prompts)

        accs = []
        for k, generated_text in enumerate(generated_texts):
            accs.append("yes" in generated_text.lower())
            # pred = generated_text.replace("Answer:", "").split(".")[0].strip()
            # pred = int(pred) - 1 if pred.isdigit() else -1
            # accs.append(pred == i)

            results.append({
                "stimuli": stimuli,
                "sentence": sentences[k],
                "generated_text": generated_text,
                "valid": "yes" in generated_text.lower(),
            })

        acc = np.mean(accs)

        print(f"Accuracy for stimuli={stimuli}: {acc:.4f}")

    write_json(results, f"{SAVE_DIR}/{model_name}/ablation/similarity_no_ablation_results_v4.json")

