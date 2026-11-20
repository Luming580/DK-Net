"""Evaluate a DK-Net checkpoint on the Synapse CT test volumes."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import DataLoader
from tqdm import tqdm

from dknet import DKNet
from dknet.data import SynapseTestDataset
from dknet.metrics import synapse_class_metrics
from dknet.runtime import load_model_weights, resolve_device, set_seed, write_json


CLASS_NAMES = ("Aorta", "Gallbladder", "Left Kidney", "Right Kidney", "Liver", "Pancreas", "Spleen", "Stomach")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/Synapse/test_vol_h5"), help="Directory containing Synapse test .npy.h5 files.")
    parser.add_argument("--list-file", type=Path, default=Path("lists/synapse/test_vol.txt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/synapse/final_model.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/synapse_test"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--normalize-input", action="store_true", help="Use (x - 0.5) / 0.5, matching the legacy inference script.")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0, help="Debug only: limit the number of evaluated volumes.")
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
    return logging.getLogger("test_synapse")


@torch.inference_mode()
def predict_volume(
    image: np.ndarray,
    model: torch.nn.Module,
    image_size: int,
    device: torch.device,
    normalize_input: bool,
) -> np.ndarray:
    prediction = np.zeros(image.shape, dtype=np.int16)
    for index, image_slice in enumerate(image):
        height, width = image_slice.shape
        resized = image_slice
        if (height, width) != (image_size, image_size):
            resized = zoom(image_slice, (image_size / height, image_size / width), order=3)
        if normalize_input:
            resized = (resized - 0.5) / 0.5
        tensor = torch.from_numpy(np.ascontiguousarray(resized)).unsqueeze(0).unsqueeze(0).float().to(device)
        output = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()
        if (height, width) != (image_size, image_size):
            output = zoom(output, (height / image_size, width / image_size), order=0)
        prediction[index] = output.astype(np.int16)
    return prediction


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.data_root.is_dir() or not args.list_file.is_file() or not args.checkpoint.is_file():
        missing = [str(path) for path in (args.data_root, args.list_file, args.checkpoint) if not path.exists()]
        raise FileNotFoundError(", ".join(missing))
    if args.num_classes != len(CLASS_NAMES) + 1:
        raise ValueError(f"This Synapse evaluator expects {len(CLASS_NAMES) + 1} classes.")

    logger = configure_logging(args.output_dir)
    set_seed(args.seed)
    device = resolve_device(args.device)
    logger.info("Device: %s", device)
    dataset = SynapseTestDataset(args.data_root, args.list_file)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = DKNet(num_classes=args.num_classes).to(device)
    load_model_weights(model, args.checkpoint, device)
    model.eval()

    prediction_dir = args.output_dir / "predictions" if args.save_predictions else None
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_case: list[np.ndarray] = []
    case_rows: list[dict[str, object]] = []
    for case_index, batch in enumerate(tqdm(loader, desc="Testing", ncols=110), start=1):
        if args.max_cases and case_index > args.max_cases:
            break
        image = batch["image"].squeeze(0).numpy()
        target = batch["label"].squeeze(0).numpy()
        prediction = predict_volume(image, model, args.image_size, device, args.normalize_input)
        class_metrics = np.asarray(
            [synapse_class_metrics(prediction == class_id, target == class_id) for class_id in range(1, args.num_classes)],
            dtype=np.float64,
        )
        metrics_by_case.append(class_metrics)
        case_name = batch["case_name"][0]
        case_rows.append(
            {
                "case_name": case_name,
                "mean_dice": float(class_metrics[:, 0].mean()),
                "mean_hd95": float(class_metrics[:, 1].mean()),
            }
        )
        if prediction_dir is not None:
            np.savez_compressed(prediction_dir / f"{case_name}.npz", prediction=prediction)

    if not metrics_by_case:
        raise RuntimeError("No test volumes were evaluated.")
    average_metrics = np.stack(metrics_by_case).mean(axis=0)
    organ_rows = [
        {
            "class_id": class_id,
            "organ": CLASS_NAMES[class_id - 1],
            "dice": float(average_metrics[class_id - 1, 0]),
            "dice_percent": float(average_metrics[class_id - 1, 0] * 100.0),
            "hd95": float(average_metrics[class_id - 1, 1]),
        }
        for class_id in range(1, args.num_classes)
    ]
    result = {
        "checkpoint": str(args.checkpoint),
        "test_volumes": len(metrics_by_case),
        "mean_dice": float(average_metrics[:, 0].mean()),
        "mean_dice_percent": float(average_metrics[:, 0].mean() * 100.0),
        "mean_hd95": float(average_metrics[:, 1].mean()),
        "normalize_input": args.normalize_input,
    }
    write_json(args.output_dir / "metrics.json", result)
    write_csv(args.output_dir / "per_case_metrics.csv", case_rows)
    write_csv(args.output_dir / "per_organ_metrics.csv", organ_rows)
    logger.info("Mean Dice: %.2f%%", result["mean_dice_percent"])
    logger.info("Mean HD95: %.4f", result["mean_hd95"])


if __name__ == "__main__":
    main()
