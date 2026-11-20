"""Train DK-Net on TN3K thyroid-nodule ultrasound images."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dknet import DKNet
from dknet.data import TN3KDataset
from dknet.losses import DiceLoss
from dknet.metrics import binary_dice, binary_jaccard
from dknet.runtime import resolve_device, save_checkpoint, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/tn3k"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tn3k"))
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ce-weight", type=float, default=0.4)
    parser.add_argument("--dice-weight", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--poly-lr", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=0, help="Debug only: limit batches per epoch.")
    parser.add_argument("--max-val-batches", type=int, default=0, help="Debug only: limit validation batches.")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(output_dir / "train.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger("train_tn3k")


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 0) -> dict[str, float]:
    model.eval()
    dice_scores: list[float] = []
    jaccard_scores: list[float] = []
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()
        predictions = model(images).argmax(dim=1).cpu().numpy()
        for prediction, label in zip(predictions, labels):
            dice_scores.append(binary_dice(prediction == 1, label == 1))
            jaccard_scores.append(binary_jaccard(prediction == 1, label == 1))
    if not dice_scores:
        raise RuntimeError("No validation samples were evaluated.")
    return {"dice": float(sum(dice_scores) / len(dice_scores)), "jaccard": float(sum(jaccard_scores) / len(jaccard_scores))}


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.save_every < 1:
        raise ValueError("epochs, batch-size, and save-every must be positive.")
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)

    logger = configure_logging(args.output_dir)
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    logger.info("Device: %s", device)
    logger.info("Arguments: %s", vars(args))

    train_dataset = TN3KDataset(args.data_root, split="train", fold=args.fold, img_size=args.image_size, augment=True)
    validation_dataset = TN3KDataset(args.data_root, split="val", fold=args.fold, img_size=args.image_size, augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = DKNet(num_classes=2).to(device)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(2)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    training_batches = min(len(train_loader), args.max_train_batches or len(train_loader))
    total_steps = max(args.epochs * training_batches, 1)
    best_dice = float("-inf")
    history_path = args.output_dir / "history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=["epoch", "lr", "loss", "ce_loss", "dice_loss", "val_dice", "val_jaccard"])
        writer.writeheader()
        step = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            totals = {"loss": 0.0, "ce_loss": 0.0, "dice_loss": 0.0}
            batches = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=110)
            for batch_index, batch in enumerate(progress, start=1):
                if args.max_train_batches and batch_index > args.max_train_batches:
                    break
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(images)
                    current_ce = ce_loss(logits, labels)
                    current_dice = dice_loss(logits, labels)
                    loss = args.ce_weight * current_ce + args.dice_weight * current_dice
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step += 1
                if args.poly_lr:
                    lr = args.lr * (1.0 - step / total_steps) ** 0.9
                    for parameter_group in optimizer.param_groups:
                        parameter_group["lr"] = lr
                batches += 1
                totals["loss"] += loss.item()
                totals["ce_loss"] += current_ce.item()
                totals["dice_loss"] += current_dice.item()
                progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

            if not batches:
                raise RuntimeError("No training batches were processed.")
            validation = evaluate(model, validation_loader, device, args.max_val_batches)
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "loss": totals["loss"] / batches,
                "ce_loss": totals["ce_loss"] / batches,
                "dice_loss": totals["dice_loss"] / batches,
                "val_dice": validation["dice"],
                "val_jaccard": validation["jaccard"],
            }
            writer.writerow(row)
            history_file.flush()
            logger.info("Epoch %d: loss=%.5f, val_dice=%.4f, val_jaccard=%.4f", epoch, row["loss"], row["val_dice"], row["val_jaccard"])
            if validation["dice"] > best_dice:
                best_dice = validation["dice"]
                save_checkpoint(args.output_dir / "best_model.pth", model, optimizer, epoch, vars(args))
            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_checkpoint(args.output_dir / f"epoch_{epoch:03d}.pth", model, optimizer, epoch, vars(args))

    save_checkpoint(args.output_dir / "final_model.pth", model, optimizer, args.epochs, vars(args))
    write_json(
        args.output_dir / "summary.json",
        {
            "best_validation_dice": best_dice,
            "best_validation_dice_percent": best_dice * 100.0,
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
        },
    )
    logger.info("Training complete. Best validation Dice: %.2f%%", best_dice * 100.0)


if __name__ == "__main__":
    main()
