
import torch
import pathlib
import json

from src.diffusion_mnist_normalize import CondDiffuser
from src.simple_unet import CondSimpleUnet
from src.ema import EMA

def build_model(model_config, device):
    model_class = model_config["model_class"]
    model_params = model_config["model_params"]
    if model_class == "CondSimpleUnet":
        model = CondSimpleUnet(
            in_ch=model_params["in_ch"],
            time_embed_dim=model_params["time_embed_dim"],
            num_labels=model_params["num_labels"],
        ).to(device)
    # elif model_class == "Dit":
    #     model = Dit(
    #         in_ch=model_params["in_ch"],
    #         time_embed_dim=model_params["time_embed_dim"],
    #         num_labels=model_params["num_labels"],
    #     ).to(device)
    else:
        raise ValueError("Invalid model class")
    return model

def build_diffuser(model, diffuser_config, device):
    return CondDiffuser(
        model=model,
        num_timesteps=diffuser_config["num_timesteps"],
        beta_start=diffuser_config["beta_start"],
        beta_end=diffuser_config["beta_end"],
        gamma=diffuser_config["gamma"],
        beta_schedule_type=diffuser_config["beta_schedule_type"],
        device=device,
    )

def load_config(model_name):
    config_folder_dir = pathlib.Path(f"./models/{model_name}/config")
    with open(pathlib.Path(config_folder_dir / "model_config.json"), "r") as f:
        model_config = json.load(f)
    with open(pathlib.Path(config_folder_dir / "diffuser_config.json"), "r") as f:
        diffuser_config = json.load(f)
    with open(pathlib.Path(config_folder_dir / "train_config.json"), "r") as f:
        train_config = json.load(f)
    return model_config, diffuser_config, train_config


def load_model(model, optimizer, ema: EMA, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    ema.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    return start_epoch