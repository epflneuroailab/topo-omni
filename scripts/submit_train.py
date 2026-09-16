import os
import torch
import argparse 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for TopoOmni model.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()

    command = f"bash scripts/train.sh {args.config} {num_gpus} accelerate.yml"
    if args.debug:
        command += " --debug"

    os.system(command)
