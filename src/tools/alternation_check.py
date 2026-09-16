"""Does ALTERNATING image and video steps fit, and at which image-batch-size?

The sampler interleaves a 96-image step (short seq, many images) with a 4-clip step
(long seq, big audio+video towers). Those have very different allocation shapes, and the
caching allocator does not hand memory back between them. A size that passes in isolation
can still OOM once the two alternate, which is what a real run does every other step.

Run: python src/tools/alternation_check.py [image_batch_sizes...]
"""
import glob, json, os, sys
import torch
sys.path.insert(0, __file__.rsplit("/src/", 1)[0])  # repo root
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig
from src.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as SHEET
from src.training.data_utils import DataCollatorForQwenOmni, load_and_format_dataset
from src.training.batching import split_by_modality
import yaml

SIZES = [int(a) for a in sys.argv[1:] if a.isdigit()] or [96, 64, 48]
FLUSH = "--flush" in sys.argv   # mirror train.py's empty_cache on switch
cfgy = yaml.safe_load(open("src/configs/experiments/train/continuous_4096_images.yml"))
VBS, SIDE = cfgy["train"]["batch-size"], cfgy["image-side"]

json.dump(json.load(open("data/imagenet_images.json"))[:256], open("/tmp/alt_img.json", "w"))
json.dump(json.load(open("data/koala_final_distilled.json"))[:16], open("/tmp/alt_vid.json", "w"))
proc = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
proc.tokenizer.padding_side = "right"
ds = load_and_format_dataset(["/tmp/alt_vid.json", "/tmp/alt_img.json"], proc, image_side=SIDE)
v_idx, i_idx = split_by_modality(ds)

cfg = Qwen2_5OmniThinkerConfig.from_pretrained("Qwen/Qwen2.5-Omni-3B")
cfg.position_dir = "src/" + cfgy["topo-params"]["position-dir"]
n = cfgy["topo-params"]["neighborhood"]
cfg.neighborhood = {"mode": n["mode"], "radius_min": n["radius-min"], "radius_max": n["radius-max"],
    "schedule": n["schedule"], "neighborhoods_per_batch": n["neighborhoods-per-batch"],
    "coords_seed": n["coords-seed"], "drop_dead_units": n["drop-dead-units"],
    "scale_balance": n["scale-balance"], "stratify_by_depth": n["stratify-by-depth"],
    "center_allocation": n["center-allocation"], "center_reuse": 1,
    "max_neighborhood_units": n["max-neighborhood-units"]}
cfg.activation_decay, cfg.alpha, cfg.accum = (cfgy["topo-params"]["activation-decay"],
                                              cfgy["topo-params"]["alpha"],
                                              cfgy["topo-params"]["accum"])
cfg.spatial_time_chunk = cfgy["topo-params"]["audio-chunk"]
cfg.use_cache = False; cfg.is_training = True; cfg.apply_spatial_loss = True
for sub in ("audio_config", "vision_config", "text_config"):
    getattr(cfg, sub).is_training = True
    getattr(cfg, sub).cortical_dim = SHEET.cortical_dim
cks = sorted(glob.glob("src/ckpts/topo-omni-continuous-4096/checkpoint-*"),
             key=lambda q: int(q.rsplit("-", 1)[1]))
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    cks[-1], torch_dtype=torch.bfloat16, config=cfg).to("cuda")
model.load_positions()
for nm, p in model.named_parameters(): p.requires_grad = "cortical_adaptor" in nm
model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.train()
coll = DataCollatorForQwenOmni(processor=proc)
print(f"checkpoint {cks[-1]}, alloc_conf={os.environ.get('PYTORCH_ALLOC_CONF')}", flush=True)


def one(rows):
    b = coll([ds[i] for i in rows])
    b = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in b.items() if k != "modality"}
    out = model(**b, return_dict=True)
    T = out.unified_sheet.shape[0]
    out.loss.backward()
    model.zero_grad(set_to_none=True)
    del out, b
    return T


# BASELINE FIRST: can this card do a video step at all, with no image work involved?
# If not, every "alternating" failure below is the card, not the image pipeline.
for vb in (1, 2, VBS):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        one(v_idx[:vb])
        print(f"  VIDEO-ONLY batch {vb}: OK   peak "
              f"{torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
    except torch.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        print(f"  VIDEO-ONLY batch {vb}: OOM  (no images involved -- this card cannot "
              f"run the configured video step)", flush=True)
    torch.cuda.empty_cache()

for ibs in SIZES:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ok, peak, note = True, 0.0, ""
    try:
        for cycle in range(3):
            one(i_idx[cycle * ibs % 128:][:ibs])      # image step
            if FLUSH:
                torch.cuda.empty_cache()              # what train.py does on a switch
            one(v_idx[:VBS])                          # video step, immediately after
            if FLUSH:
                torch.cuda.empty_cache()
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
    except torch.OutOfMemoryError as e:
        ok, note = False, "OOM on " + ("video" if "video" in str(e) else "a step")
        model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    print(f"  image-batch-size {ibs:>3} alternating with video x{VBS}: "
          f"{'OK  ' if ok else 'FAIL'} peak {peak:.1f} GiB {note}", flush=True)
