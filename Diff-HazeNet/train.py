import argparse
import os
import yaml
import torch
import torch.utils.data
import numpy as np

import dataload
from model import DenoisingDiffusion


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Patch-Based Denoising Diffusion Models')
    parser.add_argument('--resume', default=r'', type=str, help='reload')
    parser.add_argument("--sampling_timesteps", type=int, default=25, help="")
    parser.add_argument("--image_folder", default='results/train_out/', type=str, help="")
    parser.add_argument('--seed', default=61, type=int, metavar='N', help='')

    parser.add_argument("--config", default="allhaze.yml", type=str, help="config")
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace




def main():
    args, config = parse_args_and_config()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True
    DATASET = dataload.__dict__[config.data.dataset](config)
    diffusion = DenoisingDiffusion(args, config)
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
