"""Final gate: the REAL manifests, the REAL config, through the REAL sampler + forward."""
import json, os, sys, yaml
import numpy as np, torch
sys.path.insert(0, __file__.rsplit("/src/", 1)[0])  # repo root
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig
from src.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as SHEET
from src.training.data_utils import DataCollatorForQwenOmni, load_and_format_dataset
from src.training.batching import ModalityBatchSampler, split_by_modality

fails = []
def check(c, label, detail=""):
    print(f"{'  PASS' if c else '  FAIL'}  {label}   {detail}", flush=True)
    if not c: fails.append(label)

cfgy = yaml.safe_load(open("src/configs/experiments/train/continuous_4096_images.yml"))
IBS = cfgy["train"]["image-batch-size"]
VBS = cfgy["train"]["batch-size"]
SIDE = cfgy["image-side"]
WORLD = 4
print(f"config: image-batch-size={IBS} batch-size={VBS} image-side={SIDE} "
      f"ratio={cfgy['train']['image-step-ratio']}")

# subsample the manifests so tokenisation is quick; the pipeline code path is identical
subs = []
for m in cfgy["image-data-path"]:
    rows = json.load(open(m))
    sub = f"/tmp/fc_{os.path.basename(m)}"
    json.dump(rows[:200], open(sub, "w")); subs.append(sub)
    print(f"  {m}: {len(rows):,} rows -> sampling 200")
vid = json.load(open("data/koala_final_distilled.json"))[:32]
json.dump(vid, open("/tmp/fc_videos.json", "w"))

proc = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
proc.tokenizer.padding_side = "right"
ds = load_and_format_dataset(["/tmp/fc_videos.json"] + subs, proc, image_side=SIDE)
v_idx, i_idx = split_by_modality(ds)
check(len(v_idx) == 32 and len(i_idx) == 400, "3 manifests concatenate + split",
      f"{len(v_idx)} video / {len(i_idx)} image")

smp = ModalityBatchSampler(v_idx, i_idx, VBS, IBS, seed=42, num_replicas=WORLD,
                           image_step_ratio=cfgy["train"]["image-step-ratio"])
b = list(smp)
mod = {i: ("image" if i in set(i_idx) else "video") for i in range(len(ds))}
check(all(len({mod[j] for j in x}) == 1 for x in b), "batches homogeneous")
runs = [b[i:i+WORLD] for i in range(0, len(b), WORLD)]
check(all(len({mod[x[0]] for x in run}) == 1 for run in runs), "ranks agree per step",
      f"{len(runs)} steps")

cfg = Qwen2_5OmniThinkerConfig.from_pretrained("Qwen/Qwen2.5-Omni-3B")
cfg.position_dir = "src/" + cfgy["topo-params"]["position-dir"]
n = cfgy["topo-params"]["neighborhood"]
cfg.neighborhood = {"mode": n["mode"], "radius_min": n["radius-min"],
    "radius_max": n["radius-max"], "schedule": n["schedule"],
    "neighborhoods_per_batch": n["neighborhoods-per-batch"], "coords_seed": n["coords-seed"],
    "drop_dead_units": n["drop-dead-units"], "scale_balance": n["scale-balance"],
    "stratify_by_depth": n["stratify-by-depth"], "center_allocation": n["center-allocation"],
    "center_reuse": 1, "max_neighborhood_units": n["max-neighborhood-units"]}
cfg.activation_decay = cfgy["topo-params"]["activation-decay"]
cfg.alpha = cfgy["topo-params"]["alpha"]; cfg.accum = cfgy["topo-params"]["accum"]
cfg.spatial_time_chunk = cfgy["topo-params"]["audio-chunk"]
cfg.use_cache = False; cfg.is_training = True; cfg.apply_spatial_loss = True
for sub in ("audio_config", "vision_config", "text_config"):
    getattr(cfg, sub).is_training = True
    getattr(cfg, sub).cortical_dim = SHEET.cortical_dim
# The training run uses save_total_limit=1, so each new checkpoint DELETES the previous
# one -- checkpoint-69 was already gone by the time this was written. Discover whatever
# is on disk rather than pinning a number that will not survive the week.
import glob as _glob
_cks = sorted(_glob.glob("src/ckpts/topo-omni-continuous-4096/checkpoint-*"),
              key=lambda q: int(q.rsplit("-", 1)[1]))
if not _cks:
    raise SystemExit("no checkpoint under src/ckpts/topo-omni-continuous-4096/")
CKPT = os.environ.get("CHECK_CKPT", _cks[-1])
print(f"checkpoint: {CKPT}")
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, config=cfg).to("cuda")
model.load_positions()
for nm, pr in model.named_parameters(): pr.requires_grad = "cortical_adaptor" in nm
model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.train()
coll = DataCollatorForQwenOmni(processor=proc)

def step(label, rows, expect_T=None):
    bt = coll([ds[i] for i in rows])
    bt = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in bt.items() if k != "modality"}
    out = model(**bt, return_dict=True)
    T, sl, tl = out.unified_sheet.shape[0], out.spatial_loss, out.task_loss
    out.loss.backward()
    gn = float(torch.sqrt(sum((p.grad.float()**2).sum() for nm, p in model.named_parameters()
                              if p.grad is not None and "cortical_adaptor" in nm)))
    model.zero_grad(set_to_none=True)
    ok = np.isfinite(sl) and np.isfinite(tl) and gn > 0 and (expect_T is None or T == expect_T)
    check(ok, label, f"T={T} spatial={sl:.4f} task={tl:.4f} |grad|={gn:.3e} "
                     f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    del out, bt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

print(f"\nreal image steps at the configured batch size ({IBS}/rank):", flush=True)
imagenet = [i for i in i_idx if "imagenet-224" in json.loads(ds[i]["conversation_json"])[0]["content"][0]["image"]]
dtd = [i for i in i_idx if "dtd-224" in json.loads(ds[i]["conversation_json"])[0]["content"][0]["image"]]
check(len(imagenet) == 200 and len(dtd) == 200, "both image sources present",
      f"{len(imagenet)} imagenet / {len(dtd)} dtd")
step(f"ImageNet batch {IBS}", imagenet[:IBS], expect_T=IBS)
step(f"DTD batch {IBS}", dtd[:IBS], expect_T=IBS)
step(f"mixed-source image batch {IBS}", (imagenet[:IBS//2] + dtd[:IBS//2]), expect_T=IBS)
print(f"\nvideo step (T tracks clip duration):", flush=True)
step(f"video batch {VBS}", v_idx[:VBS])

print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILURES: ' + ', '.join(fails)}", flush=True)
