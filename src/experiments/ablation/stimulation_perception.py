import os
import json
import torch
import numpy as np
import pandas as pd

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

def plot_barplot(results, save_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    sns.set_theme(style="whitegrid", font_scale=1.5)

    df = pd.DataFrame(results)
    props = (df.groupby(["stimuli", "localizer"])["prediction"].value_counts(normalize=True).rename("proportion").reset_index())

    props["stimuli"] = props["stimuli"].apply(lambda x: x.capitalize())
    props["localizer"] = props["localizer"].apply(lambda x: x.capitalize())
    props["prediction"] = props["prediction"].apply(lambda x: x.capitalize())

    hue_order = ["Faces", "Bodies", "Scenes", "Objects", "Other"]


    g = sns.catplot(
        kind="bar",
        data=props,
        col="stimuli",
        x="localizer",
        y="proportion",
        hue="prediction",
        palette="Set2",
        legend=False,
        col_order=["Faces", "Bodies", "Scenes", "Objects"],
        order=["Faces", "Bodies", "Scenes", "Objects"],
        hue_order=hue_order,
    )

    palette = sns.color_palette("Set2", len(hue_order))
    handles = [Patch(facecolor=c, label=l) for c, l in zip(palette, hue_order)]

    g.set_axis_labels("Region to Drive", "Perceived Stimuli Proportion")
    g.set_titles("Stimuli: {col_name}")

    g.figure.legend(
        handles=handles,
        loc="lower center",
        ncol=len(hue_order),  # all items in one row
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        title="Perceived Stimuli",
    )

    g.figure.subplots_adjust(bottom=0.25)  # make room for the legend

    plt.savefig(save_path, bbox_inches="tight")
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
        outputs = model.generate(**inputs, max_new_tokens=5)
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
        "a human face",
        "a human body part that is not a face",
        "an indoor or outdoor location",
        "a toy or object",
    ]

    STIMULATE = True
    
    # model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")

    model, processor, tokenizer = load_model("Qwen/Qwen2.5-Omni-3B")
    # prompt_template = "Does the following sentence describe {stimuli}?\nAnswer only with Yes or No.\n\nSentence: {sentence}\nAnswer:"
    prompt_template = "What does the following sentence describe?\nOptions: {options}\n\nSentence: {sentence}\nAnswer:"

    model_name = "topo-omni"
    path_template = "ablation/ablation_stimuli={stimuli}_localizer={localizer}_perc={perc}_stimulate={stimulate}_v3"

    percentage = 5

    results = []
    for i, stimuli in enumerate(categories):
    
        for j, localizer in enumerate(categories):
            
            path = path_template.format(stimuli=stimuli, localizer=localizer, perc=f"{percentage:.2f}", stimulate=STIMULATE)
            path = f"{SAVE_DIR}/{model_name}/{path}.json"
            if not os.path.exists(path):
                print(f"File {path} does not exist. Skipping.")
                continue

            data = read_json(path)["responses"]

            sentences = [item["output"] for item in data]

            options = '\n' + "\n".join([f"{k+1}. {desc}" for k, desc in enumerate(descriptions)])
            prompts = [prompt_template.format(options=options, sentence=sentence) for sentence in sentences]
            generated_texts = generate_text(model, processor, tokenizer, prompts)

            for k, generated_text in enumerate(generated_texts):
                
                pred = generated_text.replace("Answer:", "").split(".")[0].strip()
                pred = int(pred) - 1 if pred.isdigit() else -1

                results.append({
                    "stimuli": stimuli,
                    "localizer": localizer,
                    "sentence": sentences[k],
                    "generated_text": generated_text,
                    "prediction": categories[pred] if 0 <= pred < len(categories) else "other",
                })


            write_json(results, f"{SAVE_DIR}/{model_name}/ablation/similarity_ablation_results_top{percentage}_stimulate={STIMULATE}_v3.json")

    plot_barplot(results, f"{SAVE_DIR}/{model_name}/ablation/stimulation_barplot_top{percentage}_stimulate={STIMULATE}_v3.png")