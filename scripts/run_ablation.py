#!/usr/bin/env python
"""
Ablation study runner for DG-HMCF.

Compares DG-HMCF against four baseline models and generates a
comparison table + visualisations.

Usage:
    python scripts/run_ablation.py \
        --config configs/daic_woz_config.yaml \
        --dataset daic_woz \
        --checkpoint outputs/run_01/checkpoints/best_model.pt \
        --output_dir outputs/ablation \
        --device cuda
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from data.utils.data_utils import collate_fn
from models.dg_hmcf import DGHMCF
from evaluation.ablation import AblationStudy
from utils.logger import Logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DG-HMCF ablation study")
    parser.add_argument("--config", type=str, default="configs/daic_woz_config.yaml")
    parser.add_argument("--dataset", type=str, default="daic_woz",
                        choices=["daic_woz", "modma", "pdch"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Trained DG-HMCF checkpoint (optional; skips re-training)")
    parser.add_argument("--output_dir", type=str, default="outputs/ablation")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ablation_epochs", type=int, default=20,
                        help="Epochs for training baseline models")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_all_datasets(config: dict, dataset_name: str, data_root: str = None):
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
    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)

    logger = Logger(
        name="dg-hmcf-ablation",
        log_dir=os.path.join(args.output_dir, "logs"),
        use_wandb=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    logger.info("Device: %s", device)

    # ---- Datasets -----------------------------------------------------------
    train_ds, val_ds, test_ds = build_all_datasets(config, args.dataset, args.data_root)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)

    # ---- Load DG-HMCF if checkpoint provided --------------------------------
    dg_hmcf_model = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model_cfg = state.get("config", config).get("model", config.get("model", {}))
        dg_hmcf_model = DGHMCF(model_cfg)
        dg_hmcf_model.load_state_dict(state["model_state_dict"])
        dg_hmcf_model = dg_hmcf_model.to(device)
        logger.info("Loaded DG-HMCF from checkpoint: %s", args.checkpoint)

    # ---- Run ablation -------------------------------------------------------
    ablation = AblationStudy(config, device)
    logger.info("Running ablation study (baseline epochs=%d)...", args.ablation_epochs)

    results = ablation.run_all(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dg_hmcf_model=dg_hmcf_model,
    )

    # ---- Results table ------------------------------------------------------
    comparison_df = ablation.compare_results(results)
    logger.info("\n%s\n", comparison_df.to_string(index=False))

    table_path = os.path.join(args.output_dir, "ablation_results.csv")
    comparison_df.to_csv(table_path, index=False)
    logger.info("Ablation table saved to %s", table_path)

    # ---- Visualise comparison -----------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        if not comparison_df.empty:
            models = comparison_df["Model"].tolist()
            f1_scores = comparison_df["F1"].tolist()
            auc_scores = comparison_df["AUROC"].tolist()

            colors = ["#C44E52" if "DG-HMCF" in m else "#4C72B0" for m in models]

            axes[0].barh(models, f1_scores, color=colors, alpha=0.85,
                         edgecolor="black", linewidth=0.8)
            axes[0].set_xlabel("F1 Score", fontsize=12)
            axes[0].set_title("F1 Score Comparison", fontsize=13, fontweight="bold")
            axes[0].set_xlim(0, 1.05)
            axes[0].grid(axis="x", alpha=0.35)

            axes[1].barh(models, auc_scores, color=colors, alpha=0.85,
                         edgecolor="black", linewidth=0.8)
            axes[1].set_xlabel("AUROC", fontsize=12)
            axes[1].set_title("AUROC Comparison", fontsize=13, fontweight="bold")
            axes[1].set_xlim(0, 1.05)
            axes[1].grid(axis="x", alpha=0.35)

        plt.suptitle("DG-HMCF Ablation Study", fontsize=15, fontweight="bold")
        plt.tight_layout()

        fig_path = os.path.join(args.output_dir, "ablation_comparison.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        logger.info("Comparison figure saved to %s", fig_path)
    except Exception as e:
        logger.warning("Could not generate comparison figure: %s", e)

    logger.info("Ablation study complete.")
    logger.finish()


if __name__ == "__main__":
    main()
