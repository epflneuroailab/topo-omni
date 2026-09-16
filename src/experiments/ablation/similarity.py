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

def generate_text(model, processor, tokenizer, prompts, max_new_tokens=2):
    """max_new_tokens=2 fits the Yes/No judge, and is why the ablation numbers stand.
    It does NOT fit the multiple-choice judge: "3. an indoor or outdoor location" is far
    longer than 2 tokens, so 194 of 792 answers came back as a bare "Answer:" or a
    truncated "The sentence" and scored as no-percept. Choice mode passes a larger value.
    """
    processor.padding_side = "left"
    conversations = [[{'role': 'user', 'content': [{"type": "text", "text": prompt}]}] for prompt in prompts]
    text = processor.apply_chat_template(conversations, add_generation_prompt=True)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    outputs = [output[output.index("Answer:\nassistant\n") + len("Answer:\nassistant\n"):].strip()for output in outputs]
    return outputs


def parse_choice(generated_text, categories, descriptions):
    """Judge answer -> category name, or None if it names nothing.

    Accepts the numbered form ("3." / "3. an indoor or outdoor location") and the prose
    form ("The sentence describes an indoor or outdoor location"), because the judge
    produces both. None is kept distinct from a wrong answer: silently folding unparseable
    answers into category 0 would inflate whichever category sits first.
    """
    txt = generated_text.replace("Answer:", "").strip()
    head = txt.split(".")[0].strip()
    if head.isdigit():
        idx = int(head) - 1
        return categories[idx] if 0 <= idx < len(categories) else None
    low = txt.lower()
    for i, desc in enumerate(descriptions):
        if desc.lower() in low:
            return categories[i]
    # last resort: a distinctive keyword per category, longest first so "body part"
    # is tried before the generic "part"
    for i, keys in enumerate([("face", "person"), ("body part", "body"),
                              ("indoor", "outdoor", "location", "scene"),
                              ("toy", "object")]):
        if any(k in low for k in keys):
            return categories[i]
    return None


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


    import argparse
    _p = argparse.ArgumentParser(description=__doc__)
    _p.add_argument("--model_name", default="topo-omni",
                    help="run title under $SAVE_DIR holding ablation/. Was hardcoded to "
                         "the strip model, so any other run's JSONs were never read.")
    _p.add_argument("--percentages", default="10",
                    help="comma-separated ablation percentages to judge. Was the fixed "
                         "sweep [30,25,20,15,10]; a run that only produced one of those "
                         "logged 'File ... does not exist' for the other four.")
    _p.add_argument("--stimulate", action="store_true",
                    help="judge the stimulation runs instead of the ablation runs.")
    _p.add_argument("--judge_mode", choices=("yesno", "choice"), default="yesno",
                    help="yesno: 'does this describe <category>?' -> a `valid` bool, which "
                         "is what plot_barplot.py reads. choice: 'which of the four does "
                         "this describe?' -> a `prediction` category name, which is what "
                         "driving_perception.py reads (row['localizer'] == row['prediction']). "
                         "The choice variant was parked in comments; the stimulation panel "
                         "cannot be built without it.")
    _args = _p.parse_args()

    STIMULATE = _args.stimulate

    # model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")

    model, processor, tokenizer = load_model("Qwen/Qwen2.5-Omni-3B")
    prompt_template = "Does the following sentence describe {stimuli}?\nAnswer only with Yes or No.\n\nSentence: {sentence}\nAnswer:"
    # prompt_template = "What does the following sentence describe?\nOptions: {options}\n\nSentence: {sentence}\nAnswer:"
    # options = '\n' + "\n".join([f"{k+1}. {desc}" for k, desc in enumerate(descriptions)])

    model_name = _args.model_name
    # _v3 -> _v4: ablation/run.py's get_save_path appends "_v4", so the _v3 template
    # matched nothing that run.py has ever written and every cell was skipped.
    _str = ""   # kept so the f-strings below stay unchanged; see run.py on strength
    path_template = ("ablation/ablation_stimuli={stimuli}_localizer={localizer}"
                     "_perc={perc}_stimulate={stimulate}" + _str + "_v4")
    options = "\n" + "\n".join(f"{k + 1}. {d}" for k, d in enumerate(descriptions))
    choice_template = ("What does the following sentence describe?\nOptions: {options}\n\n"
                       "Sentence: {sentence}\nAnswer:")

    # int when integral: the output filename interpolates this value raw, and
    # plot_barplot.py formats its side as f"{percentage:.0f}". A float 10.0 writes
    # "...top10.0..." while the reader looks for "...top10...", so they never meet.
    percentages = [(int(float(x)) if float(x).is_integer() else float(x))
                   for x in _args.percentages.split(",") if x.strip()]

    for percentage in percentages:

        results = []
        final_results = np.ones((len(categories), len(categories))) * 50
        for i, stimuli in enumerate(categories):
        
            for j, localizer in enumerate(categories):
                
                path = path_template.format(stimuli=stimuli, localizer=localizer, perc=f"{percentage:.2f}", stimulate=STIMULATE)
                path = f"{SAVE_DIR}/{model_name}/{path}.json"
                if not os.path.exists(path):
                    print(f"File {path} does not exist. Skipping.")
                    continue

                data = read_json(path)["responses"]

                sentences = [item["output"] for item in data]

                if _args.judge_mode == "choice":
                    prompts = [choice_template.format(options=options, sentence=s)
                               for s in sentences]
                    _max_tok = 16
                else:
                    prompts = [prompt_template.format(stimuli=descriptions[i], sentence=s)
                               for s in sentences]
                    _max_tok = 2
                generated_texts = generate_text(model, processor, tokenizer, prompts,
                                                max_new_tokens=_max_tok)

                accs = []
                for k, generated_text in enumerate(generated_texts):
                    rec = {
                        "stimuli": stimuli,
                        "localizer": localizer,
                        "sentence": sentences[k],
                        "generated_text": generated_text,
                    }
                    if _args.judge_mode == "choice":
                        pred_name = parse_choice(generated_text, categories, descriptions)
                        rec["prediction"] = pred_name
                        rec["valid"] = (pred_name == categories[i])
                        accs.append(rec["valid"])
                    else:
                        rec["valid"] = "yes" in generated_text.lower()
                        accs.append(rec["valid"])
                    results.append(rec)

                final_results[j, i] = np.mean(accs) * 100
                print(f"Accuracy for stimuli={stimuli} localizer={localizer} perc={percentage}: {final_results[j, i]}")

                write_json(results, f"{SAVE_DIR}/{model_name}/ablation/"
                                    f"similarity_ablation_results_top{percentage}"
                                    f"_stimulate={STIMULATE}{_str}_v4.json")

        plot_heatmap(final_results, f"{SAVE_DIR}/{model_name}/ablation/similarity_ablation_heatmap_top{percentage}_stimulate={STIMULATE}_v4.png", categories, percentage, STIMULATE)