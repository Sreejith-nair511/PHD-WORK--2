#!/usr/bin/env python
"""
Training script for DG-HMCF.

Usage:
    python scripts/train.py --config configs/daic_woz_config.yaml \
        --dataset daic_woz --data_root data/raw/daic_woz \
        --output_dir outputs/run_01 --device cuda

Arguments:
    --config      : path to YAML configuration file
    --dataset     : dataset name (daic_woz | modma | pdch)
    --data_root   : root directory of the dataset
    --output_dir  : output directory for checkpoints and logs
    --resume      : path to checkpoint to resume from
    --device      : compute device (cpu | cuda | cuda:0 etc.)
    --seed        : random seed
    --batch_size  : override batch size from config
    --epochs      : override number of epochs from config
    --lr          : override learning rate from config
    --no_wandb    : disable WandB logging even if configured
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from data.utils.data_utils import collate_fn
from models.dg_hmcf import DGHMCF
from training.trainer import Trainer
from utils.logger import Logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DG-HMCF model")
    parser.add_argument("--config", type=str, default="configs/daic_woz_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--dataset", type=str, default="daic_woz",
                        choices=["daic_woz", "modma", "pdch"],
                        help="Dataset to use")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override dataset root directory")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Compute device (cpu | cuda | cuda:N)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable WandB logging")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and merge YAML configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # If config references base_config, load and merge
    defaults = config.pop("defaults", [])
    if isinstance(defaults, list) and "base_config" in defaults:
        base_path = os.path.join(os.path.dirname(config_path), "base_config.yaml")
        if os.path.exists(base_path):
            with open(base_path, "r", encoding="utf-8") as f:
                base_config = yaml.safe_load(f)
            # Shallow merge: dataset config overrides base
            merged = base_config.copy()
            for key, val in config.items():
                if isinstance(val, dict) and key in merged and isinstance(merged[key], dict):
                    merged[key] = {**merged[key], **val}
                else:
                    merged[key] = val
            return merged

    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_datasets(config: dict, dataset_name: str, data_root: str = None):
    """Instantiate train / val / test datasets."""
    ds_cfg = config.get("dataset", {})
    root = data_root or ds_cfg.get("root_dir", f"data/raw/{dataset_name}")
    phq8_thresh = ds_cfg.get("phq8_threshold", 10)
    modalities = ds_cfg.get("modalities", None)

    if dataset_name == "daic_woz":
        from data.datasets.daic_woz_dataset import DAICWOZDataset
        train_ds = DAICWOZDataset(root, split="train", phq8_threshold=phq8_thresh,
                                   modalities=modalities, augment=True)
        val_ds = DAICWOZDataset(root, split="dev", phq8_threshold=phq8_thresh,
                                 modalities=modalities)
        test_ds = DAICWOZDataset(root, split="test", phq8_threshold=phq8_thresh,
                                  modalities=modalities)

    elif dataset_name == "modma":
        from data.datasets.modma_dataset import MODMADataset
        train_ds = MODMADataset(root, split="train", phq8_threshold=phq8_thresh,
                                 modalities=modalities, augment=True)
        val_ds = MODMADataset(root, split="val", phq8_threshold=phq8_thresh,
                               modalities=modalities)
        test_ds = MODMADataset(root, split="test", phq8_threshold=phq8_thresh,
                                modalities=modalities)

    elif dataset_name == "pdch":
        from data.datasets.pdch_dataset import PDCHDataset
        train_ds = PDCHDataset(root, split="train", phq8_threshold=phq8_thresh,
                                modalities=modalities, augment=True)
        val_ds = PDCHDataset(root, split="val", phq8_threshold=phq8_thresh,
                              modalities=modalities)
        test_ds = PDCHDataset(root, split="test", phq8_threshold=phq8_thresh,
                               modalities=modalities)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_ds, val_ds, test_ds


def main() -> None:
    args = parse_args()

    # ---- Load config --------------------------------------------------------
    config = load_config(args.config)

    # Apply CLI overrides
    tr_cfg = config.setdefault("training", {})
    if args.batch_size is not None:
        tr_cfg["batch_size"] = args.batch_size
    if args.epochs is not None:
        tr_cfg["epochs"] = args.epochs
    if args.lr is not None:
        tr_cfg["learning_rate"] = args.lr
    if args.no_wandb:
        config.setdefault("logging", {})["use_wandb"] = False

    seed = args.seed or tr_cfg.get("seed", 42)
    set_seed(seed)

    # ---- Setup output dirs --------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    config["logging"]["log_dir"] = os.path.join(args.output_dir, "logs")
    config["logging"]["checkpoint_dir"] = os.path.join(args.output_dir, "checkpoints")

    # ---- Logger -------------------------------------------------------------
    log_cfg = config.get("logging", {})
    experiment_logger = Logger(
        name=f"dg-hmcf-{args.dataset}",
        log_dir=log_cfg.get("log_dir", "logs"),
        use_wandb=log_cfg.get("use_wandb", False),
        wandb_project=log_cfg.get("wandb_project", "dg-hmcf"),
        wandb_config=config,
    )
    experiment_logger.log_hyperparams(tr_cfg)

    # ---- Device -------------------------------------------------------------
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    experiment_logger.info("Using device: %s", device)

    # ---- Datasets & DataLoaders --------------------------------------------
    experiment_logger.info("Loading dataset: %s", args.dataset)
    train_ds, val_ds, test_ds = build_datasets(config, args.dataset, args.data_root)
    experiment_logger.info(
        "Splits: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds)
    )

    data_cfg = config.get("data", {})
    num_workers = data_cfg.get("num_workers", 0)
    pin_memory = data_cfg.get("pin_memory", False) and (device.type == "cuda")
    batch_size = tr_cfg.get("batch_size", 16)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
    )

    # ---- Model --------------------------------------------------------------
    model = DGHMCF(config["model"])
    param_counts = model.count_parameters()
    experiment_logger.log_model_summary("DG-HMCF", param_counts["total"])
    experiment_logger.info("Parameter breakdown: %s", param_counts)

    # ---- Trainer ------------------------------------------------------------
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # Resume from checkpoint if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ---- Train --------------------------------------------------------------
    experiment_logger.info("Starting training...")
    train_result = trainer.train()

    # ---- Final test evaluation ----------------------------------------------
    from evaluation.evaluator import Evaluator
    evaluator = Evaluator(model, config, device)

    # Load best model
    best_ckpt = os.path.join(config["logging"]["checkpoint_dir"], "best_model.pt")
    if os.path.exists(best_ckpt):
        trainer.load_checkpoint(best_ckpt)

    test_metrics = evaluator.evaluate(test_loader)
    experiment_logger.info("Test metrics: %s", test_metrics)
    experiment_logger.log_metrics(test_metrics, prefix="test")

    # Save predictions
    predictions_df = evaluator.generate_predictions(test_loader)
    pred_path = os.path.join(args.output_dir, "predictions.csv")
    predictions_df.to_csv(pred_path, index=False)
    experiment_logger.info("Predictions saved to %s", pred_path)

    # ---- Visualisations -----------------------------------------------------
    from utils.visualization import (
        plot_training_curves, plot_confusion_matrix, plot_reliability_weights
    )
    import numpy as np

    viz_dir = os.path.join(args.output_dir, "figures")

    plot_training_curves(
        train_result["history"],
        save_path=os.path.join(viz_dir, "training_curves.png"),
    )
    plot_confusion_matrix(
        labels=predictions_df["true_label"].values,
        preds=predictions_df["predicted_label"].values,
        save_path=os.path.join(viz_dir, "confusion_matrix.png"),
    )

    experiment_logger.info("Training complete. Results in %s", args.output_dir)
    experiment_logger.finish()


if __name__ == "__main__":
    main()
