"""Evaluate DK-Net on a TN3K split."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dknet import DKNet
from dknet.data import TN3KDataset
from dknet.metrics import binary_dice, binary_jaccard
from dknet.runtime import load_model_weights, resolve_device, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/tn3k"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/tn3k/best_model.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tn3k_test"))
    parser.add_argument("--split", choices=("train", "val", "trainval", "test"), default="test")
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0, help="Debug only: limit evaluated batches.")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(output_dir / "test.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger("test_tn3k")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not args.data_root.is_dir() or not args.checkpoint.is_file():
        missing = [str(path) for path in (args.data_root, args.checkpoint) if not path.exists()]
        raise FileNotFoundError(", ".join(missing))
    logger = configure_logging(args.output_dir)
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = TN3KDataset(args.data_root, args.split, args.fold, args.image_size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = DKNet(num_classes=2).to(device)
    load_model_weights(model, args.checkpoint, device)
    model.eval()

    prediction_dir = args.output_dir / "predictions" if args.save_predictions else None
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    dice_scores: list[float] = []
    jaccard_scores: list[float] = []
    for batch_index, batch in enumerate(tqdm(loader, desc=f"Testing {args.split}", ncols=110), start=1):
        if args.max_batches and batch_index > args.max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].numpy()
        predictions = model(images).argmax(dim=1).cpu().numpy()
        for case_name, prediction, label in zip(batch["case_name"], predictions, labels):
            dice_scores.append(binary_dice(prediction == 1, label == 1))
            jaccard_scores.append(binary_jaccard(prediction == 1, label == 1))
            if prediction_dir is not None:
                Image.fromarray((prediction.astype(np.uint8) * 255)).save(prediction_dir / f"{case_name}.png")

    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "samples": len(dice_scores),
        "dice": float(np.mean(dice_scores)),
        "dice_percent": float(np.mean(dice_scores) * 100.0),
        "jaccard": float(np.mean(jaccard_scores)),
        "jaccard_percent": float(np.mean(jaccard_scores) * 100.0),
    }
    write_json(args.output_dir / f"{args.split}_metrics.json", result)
    logger.info("Dice: %.2f%%", result["dice_percent"])
    logger.info("Jaccard: %.2f%%", result["jaccard_percent"])


if __name__ == "__main__":
    main()
