import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


try:
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
except AttributeError:
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST


class TN3KDataset(Dataset):
    """TN3K thyroid nodule ultrasound segmentation dataset.

    Expected directory:
      tn3k/
        trainval-image/
        trainval-mask/
        test-image/
        test-mask/
        tn3k-trainval-fold0.json ... tn3k-trainval-fold4.json
    """

    def __init__(
        self,
        base_dir,
        split="train",
        fold=0,
        img_size=224,
        augment=False,
        mean=0.5,
        std=0.5,
    ):
        self.base_dir = Path(base_dir)
        self.split = split
        self.fold = int(fold)
        self.img_size = int(img_size)
        self.augment = augment
        self.mean = float(mean)
        self.std = float(std)

        if split in ("train", "val"):
            split_file = self.base_dir / f"tn3k-trainval-fold{self.fold}.json"
            with split_file.open("r", encoding="utf-8") as f:
                split_data = json.load(f)
            ids = split_data[split]
            image_dir = self.base_dir / "trainval-image"
            mask_dir = self.base_dir / "trainval-mask"
        elif split == "trainval":
            image_dir = self.base_dir / "trainval-image"
            mask_dir = self.base_dir / "trainval-mask"
            ids = sorted(int(p.stem) for p in image_dir.glob("*.jpg"))
        elif split == "test":
            image_dir = self.base_dir / "test-image"
            mask_dir = self.base_dir / "test-mask"
            ids = sorted(int(p.stem) for p in image_dir.glob("*.jpg"))
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.samples = []
        for sample_id in ids:
            stem = f"{int(sample_id):04d}"
            image_path = image_dir / f"{stem}.jpg"
            mask_path = mask_dir / f"{stem}.jpg"
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)
            self.samples.append((stem, image_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split={split} in {self.base_dir}")

    def __len__(self):
        return len(self.samples)

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        if random.random() < 0.5:
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)
        if random.random() < 0.5:
            angle = random.uniform(-20.0, 20.0)
            image = image.rotate(angle, resample=BILINEAR, fillcolor=0)
            mask = mask.rotate(angle, resample=NEAREST, fillcolor=0)
        return image, mask

    def __getitem__(self, index):
        case_name, image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        mask = Image.fromarray((np.asarray(mask) > 127).astype(np.uint8) * 255)

        if self.augment:
            image, mask = self._augment(image, mask)

        if self.img_size > 0:
            size = (self.img_size, self.img_size)
            image = image.resize(size, BILINEAR)
            mask = mask.resize(size, NEAREST)

        image = np.asarray(image, dtype=np.float32) / 255.0
        image = (image - self.mean) / self.std
        mask = (np.asarray(mask) > 127).astype(np.int64)

        sample = {
            "image": torch.from_numpy(image).unsqueeze(0),
            "label": torch.from_numpy(mask),
            "case_name": case_name,
        }
        return sample
