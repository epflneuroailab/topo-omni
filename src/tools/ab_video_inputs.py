"""Prove the VIDEO path is byte-for-byte unchanged by the image work.

Compares the tensors handed to the model -- input_ids, attention_mask, labels,
pixel_values_videos, input_features -- between the pre-change data_utils (backed up at
/tmp/data_utils_old.py) and the current one, for the same clips. If the model inputs are
identical then the forward, the sheet, T and the spatial loss are identical too, without
needing a GPU or fighting the spatial loss's neighbourhood sampling for determinism.
"""
import importlib.util, json, sys
import torch
sys.path.insert(0, __file__.rsplit("/src/", 1)[0])  # repo root
from transformers import Qwen2_5OmniProcessor

CLIPS = 3
rows = json.load(open("data/koala_final_distilled.json"))[:CLIPS]
json.dump(rows, open("/tmp/ab_videos.json", "w"))

proc = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
proc.tokenizer.padding_side = "right"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def inputs_from(module_path, tag):
    # data_utils imports are relative to src/, so keep sys.path stable and just swap file
    m = load_module(f"du_{tag}", module_path)
    ds = m.load_and_format_dataset("/tmp/ab_videos.json", proc)
    coll = m.DataCollatorForQwenOmni(processor=proc)
    return coll([ds[i] for i in range(len(ds))])


new = inputs_from("src/training/data_utils.py", "new")
old = inputs_from("/tmp/data_utils_old.py", "old")

keys = sorted(set(new) | set(old))
same = True
print(f"\n{'tensor':<26} {'old':<28} {'new':<28} identical")
for k in keys:
    a, b = old.get(k), new.get(k)
    if torch.is_tensor(a) and torch.is_tensor(b):
        eq = a.shape == b.shape and torch.equal(a, b)
        print(f"{k:<26} {str(tuple(a.shape)):<28} {str(tuple(b.shape)):<28} {eq}")
        same &= eq
    elif (a is None) != (b is None):
        print(f"{k:<26} {'present' if a is not None else '-':<28} "
              f"{'present' if b is not None else '-':<28} False")
        same = False

print(f"\nVIDEO PATH {'UNCHANGED -- every model input is identical' if same else 'CHANGED'}")
