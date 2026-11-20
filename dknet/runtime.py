"""Small runtime utilities shared by training and evaluation scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch.")
    return device


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    args: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": unwrap_model(model).state_dict(),
        "args": args,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_model_weights(model: torch.nn.Module, path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cleaned_state_dict = {
        key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()
    }
    unwrap_model(model).load_state_dict(cleaned_state_dict, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {}


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
