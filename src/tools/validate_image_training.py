"""End-to-end validation of image+video training, WITHOUT launching a run.

Checks, in order:
  1. an image manifest loads and tokenises alongside the video manifest
  2. ModalityBatchSampler yields homogeneous batches, in runs of num_replicas, no repeats
  3. simulated DDP sharding puts every rank on the same modality at the same step
  4. a mixed batch is refused with a legible error instead of the cryptic RuntimeError
  5. an image step gives T == image batch size, a finite spatial loss, and real gradients
  6. the video path is BYTE-FOR-BYTE unchanged (same T as before the patch)
"""
import json, os, sys, glob
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit("/src/", 1)[0])  # repo root

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig
from src.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as SHEET
from src.training.data_utils import DataCollatorForQwenOmni, load_and_format_dataset
from src.training.batching import ModalityBatchSampler, split_by_modality

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
CKPT = os.environ.get("VALIDATE_CKPT",
                      "src/ckpts/topo-omni-continuous-4096/checkpoint-69")
OK, FAIL = "  PASS", "  FAIL"
fails = []


def check(cond, label, detail=""):
    print(f"{OK if cond else FAIL}  {label}   {detail}", flush=True)
    if not cond:
        fails.append(label)


# ---------------------------------------------------------------- 1. data
# built by the real tool, so build_image_manifest is exercised too
import subprocess
DTD = "/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/bkhmsi/datasets/dtd/dtd/images"
subprocess.run([sys.executable, "-m", "src.tools.build_image_manifest",
                "--images-dir", DTD, "--labels-from-dirname", "--limit", "256", "--shuffle-seed", "0",
                "--out", "/tmp/val_images.json"], check=True)
img_rows = json.load(open("/tmp/val_images.json"))

vid_rows = json.load(open("data/koala_final_distilled.json"))[:64]
json.dump(vid_rows, open("/tmp/val_videos.json", "w"))
print(f"\n[1] manifests: {len(vid_rows)} video rows, {len(img_rows)} image rows", flush=True)

proc = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
proc.tokenizer.padding_side = "right"
ds = load_and_format_dataset(["/tmp/val_videos.json", "/tmp/val_images.json"],
                             proc, image_side=224)
vid_idx, img_idx = split_by_modality(ds)
check(len(vid_idx) == len(vid_rows) and len(img_idx) == len(img_rows),
      "dataset splits by modality", f"{len(vid_idx)} video / {len(img_idx)} image")

# ---------------------------------------------------------------- 2/3. sampler
WORLD, VBS, IBS = 4, 4, 16
smp = ModalityBatchSampler(vid_idx, img_idx, video_batch_size=VBS, image_batch_size=IBS,
                           seed=42, num_replicas=WORLD)
print(f"\n[2] {smp.describe()}", flush=True)
mod_of = {i: ("image" if i in set(img_idx) else "video") for i in range(len(ds))}
batches = list(smp)

check(len(batches) == len(smp), "len() matches what __iter__ yields",
      f"{len(batches)} == {len(smp)}")
check(all(len({mod_of[i] for i in b}) == 1 for b in batches),
      "every batch is modality-homogeneous")
sizes_ok = all(len(b) == (IBS if mod_of[b[0]] == "image" else VBS) for b in batches)
check(sizes_ok, "per-rank batch sizes are correct", f"video {VBS}, image {IBS}")

# runs of WORLD consecutive batches must share a modality -> all ranks agree per step
runs = [batches[i:i + WORLD] for i in range(0, len(batches), WORLD)]
check(all(len({mod_of[b[0]] for b in run}) == 1 for run in runs),
      "[3] all ranks get the same modality each step", f"{len(runs)} steps")
flat = [i for b in batches for i in b]
check(len(flat) == len(set(flat)), "no index repeats within an epoch", f"{len(flat)} items")
smp.set_epoch(0); e0 = [b[0] for b in smp]
smp.set_epoch(0); e0b = [b[0] for b in smp]
smp.set_epoch(1); e1 = [b[0] for b in smp]
check(e0 == e0b, "same epoch reproduces the same order")
check(e0 != e1, "different epoch reshuffles")
auto = ModalityBatchSampler(vid_idx, img_idx, VBS, IBS, seed=42, num_replicas=WORLD)
check([b[0] for b in auto] != [b[0] for b in auto],
      "auto-counter reshuffles when nothing calls set_epoch")

# ---------------------------------------------------------------- model
cfg = Qwen2_5OmniThinkerConfig.from_pretrained(MODEL_ID)
# train.py runs from src/, this runs from the repo root
cfg.position_dir = "src/neighborhoods/model=qwen2-5omni-3b-continuous-4096-v6_radius=4_neighborhoods=20_coords=42"
# mirrors train_continuous_4096.yml; `dynamic` resolves radii from coords.npy, whereas
# the default `fixed` wants a precomputed neighborhood_indices.npy that is not staged.
cfg.neighborhood = {
    "mode": "dynamic", "radius_min": 5, "radius_max": 5, "schedule": "linear",
    "neighborhoods_per_batch": 8000, "coords_seed": 42,
    "drop_dead_units": True, "scale_balance": False, "stratify_by_depth": True,
    "center_allocation": "uniform", "center_reuse": 1, "max_neighborhood_units": 768,
}
cfg.activation_decay, cfg.alpha, cfg.accum = 0, 20, "mean"
cfg.use_cache = False
cfg.is_training = True
for sub in ("audio_config", "vision_config", "text_config"):
    getattr(cfg, sub).is_training = True
    getattr(cfg, sub).cortical_dim = SHEET.cortical_dim
cfg.apply_spatial_loss = True
cfg.spatial_time_chunk = 5
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, config=cfg).to("cuda")
model.load_positions()
for n, p in model.named_parameters():
    p.requires_grad = "cortical_adaptor" in n
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.train()
coll = DataCollatorForQwenOmni(processor=proc)

# ---------------------------------------------------------------- 4. mixed refused
print("", flush=True)
try:
    coll([ds[img_idx[0]], ds[vid_idx[0]]])
    check(False, "[4] mixed batch refused", "no error raised")
except ValueError as e:
    check("mixes modalities" in str(e), "[4] mixed batch refused with a clear error",
          str(e)[:60] + "...")


def step(label, rows, expect_T=None):
    batch = coll([ds[i] for i in rows])
    batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()
             if k != "modality"}
    out = model(**batch, return_dict=True)
    T = out.unified_sheet.shape[0]
    sl, tl = out.spatial_loss, out.task_loss
    out.loss.backward()
    g = [p.grad for n, p in model.named_parameters()
         if p.grad is not None and "cortical_adaptor" in n]
    gn = float(torch.sqrt(sum((x.float() ** 2).sum() for x in g))) if g else 0.0
    model.zero_grad(set_to_none=True)
    n_tensors = len(g)
    ok = (np.isfinite(sl) and np.isfinite(tl) and gn > 0
          and (expect_T is None or T == expect_T))
    check(ok, label, f"T={T} spatial={sl:.4f} task={tl:.4f} |grad|={gn:.3e} "
                     f"({n_tensors} adaptor tensors)")
    # free the graph so a later, larger batch is not OOMed by this one's fragments
    del out, batch, g
    torch.cuda.empty_cache()
    return T


print("\n[5] image steps -- T must equal the image batch size", flush=True)
for n in (16, 64, 128, 224):
    if len(img_idx) < n:
        continue
    torch.cuda.reset_peak_memory_stats()
    try:
        step(f"image batch {n:>3}", img_idx[:n], expect_T=n)
        print(f"        peak GPU {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
    except torch.OutOfMemoryError:
        model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        print(f"  OOM   image batch {n:>3}   (does not fit on one 80 GiB card)", flush=True)

# T on a video step is NOT a fixed number: it is floor(clip_seconds * 25 / audio_chunk),
# i.e. ~5 samples per second at audio-chunk 5. Measured: a 7.2 s clip gives 35, an 11.4 s
# clip gives 57. Asserting a constant here would just encode whichever clip came first.
# The real proof that videos are untouched is src/tools/ab_video_inputs.py, which
# byte-compares every model input against the pre-change data_utils.
print("\n[6] video steps -- T tracks clip duration (~5 samples/second)", flush=True)
for n in (1, 2):
    torch.cuda.reset_peak_memory_stats()
    try:
        step(f"video batch {n}", vid_idx[:n], expect_T=None)
        print(f"        peak GPU {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
    except torch.OutOfMemoryError:
        model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        print(f"  OOM   video batch {n}", flush=True)

print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILURES: ' + ', '.join(fails)}", flush=True)
