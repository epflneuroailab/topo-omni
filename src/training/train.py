import logging
import warnings
logging.getLogger("qwen_vl_utils").setLevel(logging.ERROR)  # or CRITICAL
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import yaml
import time
import torch
import wandb
import random
import argparse
import numpy as np
from glob import glob
from shutil import copyfile
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig

from src.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from src.training.data_utils import DataCollatorForQwenOmni, load_and_format_dataset
from src.training.batching import ModalityBatchSampler, split_by_modality
from trl import SFTTrainer, SFTConfig

from dotenv import load_dotenv

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY", None)

DATA_PATH = "../data/koala_final_distilled.json"
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_pretrained_model(model_id: str, model_config: Qwen2_5OmniThinkerConfig):
    num_attempts = 10
    for i in range(num_attempts):
        try:
            model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                config=model_config,
            )
            return model
        except Exception as e:
            print(f"Attempt {i+1}/{num_attempts} failed with error: {e}")
            if i == num_attempts - 1:
                raise e
            else:
                time.sleep(2**i)  # Exponential backoff
                print("Retrying...")

    return None

class SpatialSFTTrainer(SFTTrainer):
    #: Set by main() when the run mixes images with videos. None = the stock dataloader,
    #: so a video-only run behaves exactly as before.
    modality_batching = None

    def get_train_dataloader(self):
        """Modality-homogeneous batches, with images batched much larger than videos.

        An image contributes ONE spatial-loss sample, so T on an image step equals the
        image batch size; see src/training/batching.py for the full argument and for why images
        and videos cannot share a batch.
        """
        cfg = self.modality_batching
        if not cfg:
            return super().get_train_dataloader()

        video_idx, image_idx = split_by_modality(self.train_dataset)
        if not image_idx:
            print("> no image rows in the dataset; using the stock dataloader")
            return super().get_train_dataloader()

        sampler = ModalityBatchSampler(
            video_indices=video_idx,
            image_indices=image_idx,
            video_batch_size=self.args.per_device_train_batch_size,
            image_batch_size=cfg["image_batch_size"],
            seed=self.args.seed,
            num_replicas=max(1, self.args.world_size),
            image_step_ratio=cfg.get("image_step_ratio"),
        )
        print(f"> {sampler.describe()}")

        from torch.utils.data import DataLoader
        dl = DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        return self.accelerator.prepare(dl)

    #: modality of the previous step, for the cache flush below
    _last_was_video = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # FLUSH THE ALLOCATOR ON A MODALITY SWITCH. An image step and a video step have
        # very different allocation shapes -- 96 images at 84 sequence tokens against 4
        # clips at 3605 with the audio tower live -- and the caching allocator does not
        # return blocks between them. Measured on the A100 dev pod (80 GiB),
        # image-batch-size 96, 64 and even 48 all OOM when alternated with a 4-clip
        # step, while each fits comfortably alone.
        #
        # Training runs on H200s (140 GiB), so those absolute numbers do not transfer --
        # but they were observed to sit at ~139/140 GiB during video-only steps, so the
        # fragmentation headroom there is not obviously better. empty_cache() costs a
        # few tens of ms and only fires on a switch, so it is cheap insurance either way.
        is_video = inputs.get("input_features") is not None
        if self._last_was_video is not None and is_video != self._last_was_video:
            torch.cuda.empty_cache()
        self._last_was_video = is_video

        outputs = model(**inputs)
        loss = outputs["loss"]

        extra_logs = {}
        for k, v in outputs.items():
            if k.endswith("_loss") and k != "loss" and v is not None:
                extra_logs[k] = v #.detach().float().mean().item()

        # T, the sample count of the spatial-loss corrcoef. Logged because it is set by
        # token pooling inside the forward rather than by anything in this config, and a
        # silent collapse (see the image_processor note in main()) is otherwise invisible
        # -- the run keeps training, just on a near-rank-deficient correlation.
        sheet = outputs.get("unified_sheet") if hasattr(outputs, "get") else None
        if sheet is not None:
            extra_logs["sheet_T"] = float(sheet.shape[0])

        rank = os.environ.get('LOCAL_RANK',-1)
        if extra_logs: # and (self.state.global_step % self.args.logging_steps == 1):
            self.log(extra_logs)

        loss = loss / self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss
    

def _coarse_neighborhood_cfg(cfg):
    """Normalise the optional coarse (tile-scoped) spatial-loss term.

    Returns None when the block is absent or `enabled: false`, which is what every
    pre-existing config gets -- so for them the loss stays exactly single-term.
    """
    if not cfg or not cfg.get("enabled", False):
        return None
    return {
        "radius_min": cfg.get("radius-min", 4),
        "radius_max": cfg.get("radius-max", 32),
        "schedule": cfg.get("schedule", "linear"),
        "neighborhoods_per_batch": cfg.get("neighborhoods-per-batch", 500),
        "max_neighborhood_units": cfg.get("max-neighborhood-units", 768),
        "weight": float(cfg.get("weight", 1.0)),
        # Confine each neighbourhood to its centre's own layer tile. Default OFF: a
        # cluster spanning adjacent layers is the point of a unified sheet, and hard
        # scoping would make every tile border visible. Here as an ablation.
        "tile_scoped": bool(cfg.get("tile-scoped", False)),
    }


def main(args):

    set_seed(seed=42)
    
    with open(f"configs/{args.config}", 'r', encoding="utf-8") as file:
        config_raw = file.read()
        run_config = yaml.load(config_raw, Loader=yaml.FullLoader)

    if not args.debug or args.wandb:
        report_to = "wandb"
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project="topo-omni", name=run_config["run-title"])
    else:
        report_to = None

    run_dir = f"ckpts/{run_config['run-title']}"
    os.makedirs(run_dir, exist_ok=True)

    copyfile(f"configs/{args.config}", f"{run_dir}/config.yml")

    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"


    # 3) Dataset
    # Overridable so a single-clip diagnostic can hold the data fixed; defaults to
    # the module-level DATA_PATH so normal runs are unchanged.
    data_path = run_config.get("data-path", DATA_PATH)
    # A list of manifests is concatenated; rows carrying type "image" are batched
    # separately from the videos (see SpatialSFTTrainer.get_train_dataloader).
    image_data_path = run_config.get("image-data-path")
    if image_data_path:
        data_path = ([data_path] if isinstance(data_path, str) else list(data_path))
        data_path += ([image_data_path] if isinstance(image_data_path, str)
                      else list(image_data_path))
    print(f"> dataset: {data_path}")
    # image-side must give EVERY image the same visual token count, or the thinker's
    # pooling collapses T; see load_and_format_dataset. 224 matches the eval's gratings.
    image_side = int(run_config.get("image-side", 224))
    train_dataset = load_and_format_dataset(data_path, processor, args.debug,
                                            image_side=image_side)
    print(f"> Loaded {len(train_dataset)} training samples.")

    model_config = Qwen2_5OmniThinkerConfig.from_pretrained(MODEL_ID)

    model_config.position_dir = run_config["topo-params"]["position-dir"]
    model_config.activation_decay = run_config["topo-params"]["activation-decay"]
    model_config.alpha = run_config["topo-params"]["alpha"]
    model_config.accum = run_config["topo-params"]["accum"]
    model_config.use_cache = False
    model_config.is_training = True
    model_config.audio_config.is_training = True
    model_config.vision_config.is_training = True
    model_config.text_config.is_training = True
    model_config.apply_spatial_loss = run_config["train"]["apply-spatial-loss"]

    # neighborhood geometry for the spatial loss. `dynamic` grows the radius with
    # depth (small in early vision / audio encoder layers, radius-max from the last
    # encoder layer onwards, constant across the thinker); `fixed` uses the
    # precomputed single-radius table under position-dir.
    neighborhood_cfg = run_config["topo-params"].get("neighborhood", {}) or {}
    model_config.neighborhood = {
        "mode": neighborhood_cfg.get("mode", "fixed"),
        "radius_min": neighborhood_cfg.get("radius-min", 5),
        "radius_max": neighborhood_cfg.get("radius-max", 25),
        "schedule": neighborhood_cfg.get("schedule", "linear"),
        "neighborhoods_per_batch": neighborhood_cfg.get("neighborhoods-per-batch", 100),
        "coords_seed": neighborhood_cfg.get("coords-seed", 42),
        # gradient-balance options; see spatial_utils.spatial_loss_fn
        "drop_dead_units": neighborhood_cfg.get("drop-dead-units", False),
        "scale_balance": neighborhood_cfg.get("scale-balance", False),
        "stratify_by_depth": neighborhood_cfg.get("stratify-by-depth", False),
        "center_allocation": neighborhood_cfg.get("center-allocation", "uniform"),
        # 1 = redraw centres every forward (original). Set to grad-accum to hold one draw
        # for a whole accumulation group.
        "center_reuse": neighborhood_cfg.get("center-reuse", 1),
        "max_neighborhood_units": neighborhood_cfg.get("max-neighborhood-units", None),
        # Optional SECOND spatial-loss term: a large radius confined to each layer's own
        # tile. The keys above stay the fine, unscoped term. Both are wanted because the
        # map's wavelength is ~2.4x the neighborhood radius (measured), so a single small
        # radius can only ever produce small clusters -- while a tile-wide radius ALONE
        # would drop every cross-tile pair and let adjacent layers organize independently.
        "coarse": _coarse_neighborhood_cfg(neighborhood_cfg.get("coarse", {}) or {}),
    }

    # Tokens pooled into one spatial-loss sample; see  in the thinker
    # forward. 50 is the original behaviour; smaller means more samples for the unit x unit
    # correlation the loss is built on.
    # The adaptors are constructed with the model, so the sheet layout has to set their
    # width BEFORE from_pretrained. None leaves them square (each layer's own hidden size).
    from src.models.spatial_utils import DEFAULT_SHEET_LAYOUT as _SHEET
    if _SHEET.cortical_dim is not None:
        for _sub in ("audio_config", "vision_config", "text_config"):
            getattr(model_config, _sub).cortical_dim = _SHEET.cortical_dim
        print(f"> adaptor width (cortical_dim) = {_SHEET.cortical_dim}, "
              f"sheet {_SHEET.n_row}x{_SHEET.n_col} = {_SHEET.num_units:,} units"
              + (f", {_SHEET.mm_per_px} mm/px" if _SHEET.mm_per_px else ""))

    model_config.spatial_time_chunk = run_config["topo-params"].get("audio-chunk", 50)

    cortical_init_epsilon = run_config["topo-params"]["cortical-init"]
    init_identity = run_config["topo-params"].get("identity-init", False)

    model = load_pretrained_model(MODEL_ID, model_config)
    model.init_cortical_layers(epsilon=cortical_init_epsilon, identity=init_identity)
    if model_config.apply_spatial_loss:
        model.load_positions()

    # make sure only cortical adaptor parameters are trainable
    for name, param in model.named_parameters():
        if "cortical_adaptor" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Count number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"> # Trainable Parameters: {num_params:,} / {total_params:,} ({100 * num_params / total_params:.2f}%)")

    # 5) Training config
    training_args = SFTConfig(
        output_dir=run_dir,
        num_train_epochs=run_config["train"]["n-epochs"],
        per_device_train_batch_size=run_config["train"]["batch-size"], 
        gradient_accumulation_steps=run_config["train"]["grad-accum"],
        # Default 1 keeps the old behaviour, but 1 means each save DELETES the previous
        # checkpoint: continuous-4096's checkpoint-69 was gone by the time checkpoint-138
        # existed, so an across-training comparison is only possible if every checkpoint
        # was evaluated before the next one landed. Set save-total-limit in the config
        # (null = keep all) for any run you intend to measure at several points.
        save_total_limit=run_config["train"].get("save-total-limit", 1),
        dataloader_num_workers=16 if not args.debug else 0,
        learning_rate=run_config["train"]["learning-rate"],
        bf16=True,
        gradient_checkpointing=True,
        # PREEMPTION. This is a shared cluster and a Train job can be evicted at any
        # time: topo-omni-strip-4096-images was preempted at ~step 83 of 520 by another
        # tenant's workload. train() already auto-resumes (see resume_from_checkpoint
        # below), so a preemption should cost minutes -- but with save_strategy="epoch"
        # the first checkpoint lands at step 104, so that eviction threw away the whole
        # run. Set save-steps in the config to bound the loss; keep it a divisor of the
        # steps-per-epoch so the epoch boundaries are still among the saved checkpoints.
        save_strategy=("steps" if run_config["train"].get("save-steps") else "epoch"),
        save_steps=run_config["train"].get("save-steps"),
        logging_steps=1,
        report_to=report_to,
        remove_unused_columns=False,  
        ddp_find_unused_parameters=True,
        dataloader_drop_last=False,
        dataloader_pin_memory=False,
        max_length=16_384, 
        lr_scheduler_type=run_config["train"]["lr-scheduler"],
        warmup_ratio=run_config["train"]["warmup-ratio"],
        weight_decay=run_config["train"]["weight-decay"],
        # Image and video steps have DIFFERENT batch sizes, and accelerate's
        # BatchSamplerShard pads every batch to one size unless even_batches is off --
        # which would silently repeat images to fill a video-sized batch.
        #
        # Applied ONLY when image data is present. even_batches also governs how the
        # last, short batch of an epoch is handled in the video-only path, so setting it
        # unconditionally would perturb the running baseline on resume.
        **({"accelerator_config": {"even_batches": False}} if image_data_path else {}),
    )

    model.enable_input_require_grads()

    # 6) SFTTrainer
    trainer = SpatialSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForQwenOmni(processor=processor),
        processing_class=processor,
    )

    if image_data_path:
        # image-batch-size is PER RANK and is exactly T, the sample count of the
        # spatial loss corrcoef on an image step. 224 matches what the video path
        # gets at audio-chunk 5, at roughly the same sequence cost.
        trainer.modality_batching = {
            "image_batch_size": run_config["train"].get("image-batch-size", 64),
            "image_step_ratio": run_config["train"].get("image-step-ratio"),
        }

    resume_from_checkpoint = glob(f"{run_dir}/checkpoint-*")
    print(f"> Resuming from checkpoint: {bool(resume_from_checkpoint)}")

    trainer.train(resume_from_checkpoint=bool(resume_from_checkpoint))

    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str,
                        default="train.yml", help='path of config file')
    parser.add_argument('--debug',  action='store_true',
                        help='Force debug')
    parser.add_argument('--wandb',  action='store_true',
                        help='Use WANDB')
    args = parser.parse_args()
    main(args)