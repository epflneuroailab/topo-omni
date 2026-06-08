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

from models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from data_utils import DataCollatorForQwenOmni, load_and_format_dataset
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
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]

        extra_logs = {}
        for k, v in outputs.items():
            if k.endswith("_loss") and k != "loss" and v is not None:
                extra_logs[k] = v #.detach().float().mean().item()

        rank = os.environ.get('LOCAL_RANK',-1)
        if extra_logs: # and (self.state.global_step % self.args.logging_steps == 1):
            self.log(extra_logs)

        loss = loss / self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss
    

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
    train_dataset = load_and_format_dataset(DATA_PATH, processor, args.debug)
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
        save_total_limit=1,
        dataloader_num_workers=16 if not args.debug else 0,
        learning_rate=run_config["train"]["learning-rate"],
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="epoch",
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