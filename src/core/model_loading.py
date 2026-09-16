"""Single entry point for loading the Topo-Omni model + processor.

Every evaluation/extraction script loads the trained topographic model through
:func:`load_topo_omni`, so there is exactly one place that knows how the model is
fetched. By default it pulls the public release from HuggingFace
(``epfl-neuroai/topo-omni``); pass a local directory (or set ``$TOPO_OMNI_MODEL``)
to evaluate your own checkpoint.

Inference only. We set ``apply_spatial_loss=True`` so that every ``forward`` assembles
and returns the unified 309x512 cortical sheet — but the spatial loss itself (and hence
the *training-only* neighborhood/position files under ``position_dir``) is never touched,
because the loss branch runs only when ``labels`` are passed, which none of the eval
scripts do. A fresh clone therefore needs the model weights and nothing else.
"""
import os

import numpy as np

from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT
import torch

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig
from src.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

# Public release (topographic model) and the untrained multimodal base (non-topo control).
DEFAULT_MODEL = "epfl-neuroai/topo-omni"
BASELINE_MODEL = "Qwen/Qwen2.5-Omni-3B"

# Canonical output-directory labels. Results/figures live under $SAVE_DIR/<title>/..., and the
# hosted precomputed cut is organised the same way, so these are the single source of truth for
# the run-title strings that used to be hardcoded across the scripts.
MODEL_TITLE = "topo-omni"    # topographic model
BASELINE_TITLE = "qwen2_5_3b_task_7"               # non-topographic control

# Unified cortical-sheet dimensions (rows x cols). See README "Cortical Sheet".
SHEET_H, SHEET_W = DEFAULT_SHEET_LAYOUT.n_row, DEFAULT_SHEET_LAYOUT.n_col


def resolve_model_id(model=None, baseline=False):
    """Resolve the model to load: explicit arg > ``$TOPO_OMNI_MODEL`` > packaged default.

    ``baseline=True`` selects the untrained Qwen2.5-Omni-3B (the non-topographic control
    reported in the paper), which is initialised with identity cortical adaptors.
    """
    if baseline:
        return BASELINE_MODEL
    return model or os.getenv("TOPO_OMNI_MODEL", DEFAULT_MODEL)


def load_topo_omni(model=None, device="cuda", baseline=False, random_init=False,
                   dtype=None):
    """Load the Topo-Omni model, processor, and config for inference.

    Args:
        model: HuggingFace repo id or local checkpoint dir. Defaults to
            ``$TOPO_OMNI_MODEL`` or ``epfl-neuroai/topo-omni``.
        device: torch device string (``"cuda"`` / ``"cpu"``).
        baseline: load the non-topographic Qwen2.5-Omni-3B control instead.
        random_init: build the architecture from config with NO pretrained weights -- the
            analogue of TDANN's "Unoptimized" bar. Distinct from ``baseline``, which keeps
            the pretrained backbone and only neutralises the cortical adaptors.
        dtype: torch dtype; defaults to bfloat16 on CUDA, float32 on CPU.

    Returns:
        (model, processor, config) with the model in ``.eval()`` mode on ``device``.
    """
    if baseline and random_init:
        raise ValueError("baseline and random_init are mutually exclusive")
    # A random-init model has no checkpoint of its own; the base repo supplies the
    # architecture config and the processor only.
    model_id = BASELINE_MODEL if random_init else resolve_model_id(model, baseline=baseline)
    if dtype is None:
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

    kind = ("randomly initialised Qwen2.5-Omni (architecture only)" if random_init
            else "baseline Qwen2.5-Omni" if baseline else "Topo-Omni")
    print(f"> Loading {kind} model from: {model_id}")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    config = Qwen2_5OmniThinkerConfig.from_pretrained(model_id)

    config.audio_config.is_training = False
    config.vision_config.is_training = False
    config.text_config.is_training = False
    # Needed so forward() assembles the unified cortical sheet; the spatial *loss* stays off
    # at inference (no labels are passed), so no neighborhood/position files are required.
    config.apply_spatial_loss = True

    model_kwargs = dict(config=config, dtype=dtype)
    if str(device).startswith("cuda"):
        model_kwargs["device_map"] = torch.device(device)

    if random_init:
        # `_from_config` builds the architecture straight from the config and never reads
        # the checkpoint, so there is nothing to materialise on disk. It also honours
        # `dtype`, so the weights are created directly in bf16 rather than fp32-then-cast.
        model = Qwen2_5OmniThinkerForConditionalGeneration._from_config(config, dtype=dtype)
        if str(device).startswith("cuda"):
            model = model.to(device)
    else:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_id, **model_kwargs)

    if baseline or random_init:
        # No trained cortical projection: make the adaptors a pass-through. For random_init
        # this matters twice over -- the constructor seeds W with `torch.randn`, ~35x the
        # identity's scale, which would sit in the residual stream of every layer and give
        # a broken model rather than an untrained one.
        model.init_cortical_layers(epsilon=0, identity=True)

    model.to(device).eval()
    model.refresh_cortical_pinv()
    return model, processor, config


def unified_grid_coords(H=SHEET_H, W=SHEET_W):
    """Deterministic ``(H*W, 2)`` integer coordinates of the unified cortical sheet, row-major.

    Matches ``unified_sheet.reshape(-1)`` ordering, so this replaces the training-only
    ``coords.npy`` artifact for figure/analysis code that only needs to place per-unit values
    back onto the 2-D sheet.
    """
    return np.array([(i, j) for i in range(H) for j in range(W)])
