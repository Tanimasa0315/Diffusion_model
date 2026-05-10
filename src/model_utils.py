
import torch
import pathlib
import json
import re

from src.diffusion_mnist_normalize import CondDiffuser
from src.simple_unet import CondSimpleUnet, CondSimpleUnetDeep, CondSimpleUnetDeep_GN
from src.unet import CondUNet
from src.ema import EMA

CHECKPOINT_PATTERN = re.compile(r"^checkpoint_epoch_(\d+)\.pth$")

def build_model(model_config, device):
    model_class = model_config["model_class"]
    model_params = model_config["model_params"]
    if model_class == "CondSimpleUnet":
        model = CondSimpleUnet(
            in_ch=model_params["in_ch"],
            time_embed_dim=model_params["time_embed_dim"],
            num_labels=model_params["num_labels"],
            label_scale=model_params["label_scale"],
        ).to(device)
    elif model_class == "CondSimpleUnetDeep":
        model = CondSimpleUnetDeep(
            in_ch=model_params["in_ch"],
            time_embed_dim=model_params["time_embed_dim"],
            num_labels=model_params["num_labels"],
            label_scale=model_params["label_scale"],
        ).to(device)
    elif model_class == "CondSimpleUnetDeep_GN":
        model = CondSimpleUnetDeep_GN(
            in_ch=model_params["in_ch"],
            time_embed_dim=model_params["time_embed_dim"],
            num_labels=model_params["num_labels"],
            label_scale=model_params["label_scale"],
        ).to(device)
    elif model_class in ("CondUnet", "CondUNet"):
        model = CondUNet(
            in_ch=model_params["in_ch"],
            time_embed_dim=model_params["time_embed_dim"],
            num_labels=model_params["num_labels"],
            label_scale=model_params["label_scale"],
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


def load_latest_checkpoint(checkpoint_path: pathlib.Path, model, optimizer, ema):
    """
    指定されたパスから最新のチェックポイントを読み込み、モデル、オプティマイザ、EMAの状態を復元します。
    チェックポイントには、モデルの状態、オプティマイザの状態、EMAモデルの状態、損失の履歴、および最後のエポック番号が含まれている必要があります。
    Args:
        checkpoint_path (pathlib.Path): チェックポイントが保存されているディレクトリのパス。
        model (torch.nn.Module): 復元するモデルのインスタンス。
        optimizer (torch.optim.Optimizer): 復元するオプティマイザのインスタンス。
        ema (EMA): 復元するEMAのインスタンス。
    Returns:
        tuple: (start_epoch, losses, model, optimizer, ema)
        start_epoch (int): 復元されたエポック番号の次のエポックからトレーニングを再開するためのエポック番号。
        losses (list): 復元された損失の履歴。
        model (torch.nn.Module): 復元されたモデルのインスタンス。
        optimizer (torch.optim.Optimizer): 復元されたオプティマイザのインスタンス。
        ema (EMA): 復元されたEMAのインスタンス。
    """
    checkpoints = list(pathlib.Path(checkpoint_path).glob("checkpoint_epoch_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")

    def checkpoint_epoch(path: pathlib.Path) -> int:
        match = CHECKPOINT_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"Invalid checkpoint filename: {path.name}")
        return int(match.group(1))

    latest_checkpoint = max(checkpoints, key=checkpoint_epoch)
    print(f"Loading latest checkpoint from {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    ema.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
    losses = checkpoint.get("losses", [])
    start_epoch = checkpoint["epoch"] + 1
    return start_epoch, losses, model, optimizer, ema
