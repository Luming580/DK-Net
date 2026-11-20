"""Synapse multi-organ CT dataset adapters."""

from __future__ import annotations

import random
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def _resize(image: np.ndarray, label: np.ndarray, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    if (height, width) == (image_size, image_size):
        return image, label
    scale = (image_size / height, image_size / width)
    return zoom(image, scale, order=3), zoom(label, scale, order=0)


def _augment(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    choice = random.random()
    if choice < 0.5:
        rotations = np.random.randint(0, 4)
        axis = np.random.randint(0, 2)
        return (
            np.flip(np.rot90(image, rotations), axis=axis).copy(),
            np.flip(np.rot90(label, rotations), axis=axis).copy(),
        )
    if choice < 0.75:
        angle = np.random.uniform(-20.0, 20.0)
        return (
            ndimage.rotate(image, angle, order=3, reshape=False),
            ndimage.rotate(label, angle, order=0, reshape=False),
        )
    return image, label


def _read_list(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as file:
        names = [line.strip() for line in file if line.strip()]
    if not names:
        raise RuntimeError(f"No samples found in {path}.")
    return names


class SynapseTrainDataset(Dataset):
    """Training slices stored as ``.npz`` files with ``image`` and ``label`` arrays."""

    def __init__(self, data_root: str | Path, list_file: str | Path, image_size: int = 224, augment: bool = True):
        self.data_root = Path(data_root)
        self.names = _read_list(list_file)
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        name = self.names[index]
        sample_path = self.data_root / f"{name}.npz"
        if not sample_path.is_file():
            raise FileNotFoundError(sample_path)
        with np.load(sample_path) as sample:
            image = sample["image"].astype(np.float32)
            label = sample["label"].astype(np.int64)
        if self.augment:
            image, label = _augment(image, label)
        image, label = _resize(image, label, self.image_size)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).float(),
            "label": torch.from_numpy(np.ascontiguousarray(label)).long(),
            "case_name": name,
        }


class SynapseTestDataset(Dataset):
    """Test volumes stored as ``<case>.npy.h5`` with ``image`` and ``label`` arrays."""

    def __init__(self, data_root: str | Path, list_file: str | Path):
        self.data_root = Path(data_root)
        self.names = _read_list(list_file)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        name = self.names[index]
        sample_path = self.data_root / f"{name}.npy.h5"
        if not sample_path.is_file():
            raise FileNotFoundError(sample_path)
        with h5py.File(sample_path, "r") as sample:
            image = sample["image"][:].astype(np.float32)
            label = sample["label"][:].astype(np.int64)
        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
            "case_name": name,
        }
