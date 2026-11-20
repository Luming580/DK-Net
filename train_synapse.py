"""Train DK-Net on the Synapse multi-organ CT training slices."""

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
from dknet.data import SynapseTrainDataset
from dknet.losses import DiceLoss, make_class_weights
from dknet.runtime import resolve_device, save_checkpoint, set_seed, unwrap_model, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/Synapse/train_npz"), help="Directory containing Synapse train_npz files.")
    parser.add_argument("--list-file", type=Path, default=Path("lists/synapse/train.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/synapse"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ce-weight", type=float, default=0.4)
    parser.add_argument("--dice-weight", type=float, default=0.6)
    parser.add_argument("--target-class", type=int, default=None, help="Optional class index for a targeted CE-weight experiment.")
    parser.add_argument("--target-class-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a PyTorch device string.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA automatic mixed precision.")
    parser.add_argument("--data-parallel", action="store_true", help="Use all visible CUDA devices.")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=0, help="Debug only: limit batches per epoch.")
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
    return logging.getLogger("train_synapse")


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.save_every < 1:
        raise ValueError("epochs, batch-size, and save-every must be positive.")
    if args.ce_weight < 0 or args.dice_weight < 0 or args.ce_weight + args.dice_weight == 0:
        raise ValueError("CE and Dice weights must be non-negative and must not both be zero.")
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    if not args.list_file.is_file():
        raise FileNotFoundError(args.list_file)

    logger = configure_logging(args.output_dir)
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    logger.info("Device: %s", device)
    logger.info("Arguments: %s", vars(args))

    dataset = SynapseTrainDataset(args.data_root, args.list_file, args.image_size, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if not len(loader):
        raise RuntimeError("The training DataLoader is empty.")

    model: nn.Module = DKNet(num_classes=args.num_classes).to(device)
    if args.data_parallel:
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise RuntimeError("--data-parallel requires at least two CUDA devices.")
        model = nn.DataParallel(model)

    class_weights = make_class_weights(args.num_classes, args.target_class, args.target_class_weight)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    dice_loss = DiceLoss(args.num_classes)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total_steps = args.epochs * min(len(loader), args.max_train_batches or len(loader))
    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=["epoch", "lr", "loss", "ce_loss", "dice_loss"])
        writer.writeheader()

        step = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = {"loss": 0.0, "ce_loss": 0.0, "dice_loss": 0.0}
            batches = 0
            progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=110)
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
                lr = args.lr * (1.0 - step / max(total_steps, 1)) ** 0.9
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = lr
                batches += 1
                running["loss"] += loss.item()
                running["ce_loss"] += current_ce.item()
                running["dice_loss"] += current_dice.item()
                progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.6f}")

            if not batches:
                raise RuntimeError("No training batches were processed.")
            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            row.update({key: value / batches for key, value in running.items()})
            writer.writerow(row)
            history_file.flush()
            logger.info(
                "Epoch %d: loss=%.5f, ce=%.5f, dice=%.5f, lr=%.7f",
                epoch,
                row["loss"],
                row["ce_loss"],
                row["dice_loss"],
                row["lr"],
            )
            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_checkpoint(args.output_dir / f"epoch_{epoch:03d}.pth", model, optimizer, epoch, vars(args))

    save_checkpoint(args.output_dir / "final_model.pth", model, optimizer, args.epochs, vars(args))
    write_json(
        args.output_dir / "run_config.json",
        {
            "model": unwrap_model(model).__class__.__name__,
            "num_training_slices": len(dataset),
            "device": str(device),
            "target_class": args.target_class,
            "target_class_weight": args.target_class_weight,
            "ce_weight": args.ce_weight,
            "dice_weight": args.dice_weight,
        },
    )
    logger.info("Training complete. Checkpoint: %s", args.output_dir / "final_model.pth")


if __name__ == "__main__":
    main()
