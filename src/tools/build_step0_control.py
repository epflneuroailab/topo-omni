"""Materialise the UNTRAINED control as a checkpoint: the training run's step 0.

`load_topo_omni(baseline=True)` cannot serve here. It neutralises the adaptors to a
pass-through, which pins cortical_dim to the layer's hidden size (1280) -- so the sheet
comes out 33*1280 = 42,240 wide instead of 528*256 = 135,168 and the reshape throws.
An identity map cannot fill a 4096-wide sheet from 1280 hidden dims.

What training actually starts from is CorticalAdaptor.initialize(identity=True), which
TILES the identity (eye(1280).repeat(1,4)[:, :4096]) and adds epsilon*randn. That is a
well-defined 4096-wide adaptor, and it is the honest baseline for "how much of the
measured clustering did training produce": same architecture, same layout, same coords,
same adaptor width, zero optimisation steps.

    python src/tools/build_step0_control.py --out src/ckpts/strip-4096-step0
    python src/tools/build_step0_control.py --random-backbone --out src/ckpts/strip-4096-randinit
"""
import argparse, os, sys
import torch, yaml
sys.path.insert(0, __file__.rsplit("/src/", 1)[0])  # repo root
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig
from src.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as SHEET

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--train-config", default="src/configs/experiments/train/strip_4096_images.yml")
ap.add_argument("--random-backbone", action="store_true",
                help="architecture only, no pretrained weights (the absolute floor)")
args = ap.parse_args()

tc = yaml.safe_load(open(args.train_config))["topo-params"]
cfg = Qwen2_5OmniThinkerConfig.from_pretrained(MODEL_ID)
cfg.apply_spatial_loss = True
for sub in ("audio_config", "vision_config", "text_config"):
    getattr(cfg, sub).is_training = False
    getattr(cfg, sub).cortical_dim = SHEET.cortical_dim

print(f"layout {SHEET.mode} {SHEET.n_row}x{SHEET.n_col}, cortical_dim={SHEET.cortical_dim}")
if args.random_backbone:
    model = Qwen2_5OmniThinkerForConditionalGeneration._from_config(cfg, dtype=torch.bfloat16)
else:
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID, config=cfg, dtype=torch.bfloat16)

# exactly train.py's step-0 initialisation
eps = tc["cortical-init"]
ident = tc.get("identity-init", False)
print(f"initialize(epsilon={eps}, identity={ident})  "
      f"{'random backbone' if args.random_backbone else 'pretrained backbone'}",
      flush=True)
# NOT init_cortical_layers(): that ends in refresh_cortical_pinv(), which is 69 SVDs of a
# 1280x4096 matrix on CPU and ran >17 min without finishing. pinv is a forward-time cache,
# rebuilt lazily on GPU by whatever loads this checkpoint, and it is not persisted anyway.
from src.models.qwen2_5_omni import CorticalAdaptor
n = 0
for module in model.modules():
    if isinstance(module, CorticalAdaptor):
        module.initialize(epsilon=eps, identity=ident)
        n += 1
print(f"initialised {n} cortical adaptors (pinv left to the loader)", flush=True)

os.makedirs(args.out, exist_ok=True)
model.save_pretrained(args.out, safe_serialization=True)
Qwen2_5OmniProcessor.from_pretrained(MODEL_ID).save_pretrained(args.out)
print(f"wrote {args.out}")
