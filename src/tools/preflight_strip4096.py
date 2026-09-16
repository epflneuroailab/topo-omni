"""Pre-flight the strip-4096 run: build the model exactly as train.py does, from the BASE
checkpoint with fresh adaptors, and push one image batch and one video batch through it.

Catches the things that would kill a submitted job in its first minutes: the adaptor width
vs the strip reshape (33*4096 must equal encoder_rows*n_col/2), the new coords loading,
and the image path under a layout it has never run on.
"""
import json, os, sys, yaml
import numpy as np, torch
sys.path.insert(0, __file__.rsplit("/src/", 1)[0])  # repo root
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig
from src.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as S
from src.training.data_utils import DataCollatorForQwenOmni, load_and_format_dataset
from src.training.batching import split_by_modality

CFG = "src/configs/experiments/train/strip_4096_images.yml"
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
fails = []
def check(c, label, detail=""):
    print(f"{'  PASS' if c else '  FAIL'}  {label}   {detail}", flush=True)
    if not c: fails.append(label)

cfgy = yaml.safe_load(open(CFG))
print(f"layout {S.mode} {S.n_row}x{S.n_col} = {S.num_units:,} units, "
      f"cortical_dim={S.cortical_dim}, units/layer={S.units_per_layer}, "
      f"mm_per_px={S.mm_per_px}")
check(S.mode == "strip" and S.cortical_dim == 4096 and S.n_row == 816,
      "TOPO_SHEET_LAYOUT is strip-4096", f"{S.mode} {S.n_row}x{S.n_col}")
check(S.num_encoder_layers * S.cortical_dim == S.encoder_rows * (S.n_col // 2),
      "encoder band area == L * cortical_dim",
      f"{S.num_encoder_layers}*{S.cortical_dim} == {S.encoder_rows}*{S.n_col//2}")
check(S.num_thinker_layers * S.cortical_dim == (S.n_row - S.encoder_rows) * S.n_col,
      "thinker band area == L * cortical_dim")

SIDE = cfgy["image-side"]
json.dump(json.load(open("data/imagenet_images.json"))[:24], open("/tmp/pf_img.json", "w"))
json.dump(json.load(open("data/koala_final_distilled.json"))[:4], open("/tmp/pf_vid.json", "w"))
proc = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
proc.tokenizer.padding_side = "right"
ds = load_and_format_dataset(["/tmp/pf_vid.json", "/tmp/pf_img.json"], proc, image_side=SIDE)
v_idx, i_idx = split_by_modality(ds)

cfg = Qwen2_5OmniThinkerConfig.from_pretrained(MODEL_ID)
cfg.position_dir = "src/" + cfgy["topo-params"]["position-dir"]
n = cfgy["topo-params"]["neighborhood"]
cfg.neighborhood = {"mode": n["mode"], "radius_min": n["radius-min"], "radius_max": n["radius-max"],
    "schedule": n["schedule"], "neighborhoods_per_batch": n["neighborhoods-per-batch"],
    "coords_seed": n["coords-seed"], "drop_dead_units": n["drop-dead-units"],
    "scale_balance": n["scale-balance"], "stratify_by_depth": n["stratify-by-depth"],
    "center_allocation": n["center-allocation"], "center_reuse": 1,
    "max_neighborhood_units": n["max-neighborhood-units"]}
cfg.activation_decay = cfgy["topo-params"]["activation-decay"]
cfg.alpha = cfgy["topo-params"]["alpha"]; cfg.accum = cfgy["topo-params"]["accum"]
cfg.spatial_time_chunk = cfgy["topo-params"]["audio-chunk"]
cfg.use_cache = False; cfg.is_training = True; cfg.apply_spatial_loss = True
for sub in ("audio_config", "vision_config", "text_config"):
    getattr(cfg, sub).is_training = True
    getattr(cfg, sub).cortical_dim = S.cortical_dim

# exactly train.py's construction: BASE weights, fresh adaptors, identity init
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, config=cfg)
model.init_cortical_layers(epsilon=cfgy["topo-params"]["cortical-init"],
                           identity=cfgy["topo-params"].get("identity-init", False))
model.load_positions()
model = model.to("cuda")
for nm, p in model.named_parameters(): p.requires_grad = "cortical_adaptor" in nm
ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable adaptor params: {ntr:,}")
model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.train()
coll = DataCollatorForQwenOmni(processor=proc)

def step(label, rows, expect_T=None):
    b = coll([ds[i] for i in rows])
    b = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in b.items() if k != "modality"}
    out = model(**b, return_dict=True)
    T, sl, tl = out.unified_sheet.shape[0], out.spatial_loss, out.task_loss
    sheet = tuple(out.unified_sheet.shape[1:])
    out.loss.backward()
    gn = float(torch.sqrt(sum((p.grad.float()**2).sum() for nm, p in model.named_parameters()
                              if p.grad is not None and "cortical_adaptor" in nm)))
    model.zero_grad(set_to_none=True)
    ok = (sheet == (S.n_row, S.n_col) and np.isfinite(sl) and np.isfinite(tl) and gn > 0
          and (expect_T is None or T == expect_T))
    check(ok, label, f"T={T} sheet={sheet} spatial={sl:.4f} task={tl:.4f} |grad|={gn:.3e} "
                     f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    del out, b; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

print("", flush=True)
step("image batch 24", i_idx[:24], expect_T=24)
step("video batch 1", v_idx[:1])
print(f"\n{'PREFLIGHT PASSED' if not fails else 'FAILURES: ' + ', '.join(fails)}", flush=True)
