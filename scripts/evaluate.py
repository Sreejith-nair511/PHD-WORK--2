#!/usr/bin/env python
"""
Evaluation script for DG-HMCF.

Loads a trained checkpoint and evaluates it on a test split,
reporting all classification and regression metrics.

Usage:
    python scripts/evaluate.py \
        --checkpoint outputs/run_01/checkpoints/best_model.pt \
        --config configs/daic_woz_config.yaml \
        --dataset daic_woz \
        --data_root data/raw/daic_woz \
        --output_dir outputs/eval_01 \
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
from evaluation.evaluator import Evaluator
from utils.logger import Logger
from utils.visualization import (
    plot_confusion_matrix,
    plot_reliability_weights,
    plot_modality_importance,
    plot_phq8_scatter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DG-HMCF model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, default="configs/daic_woz_config.yaml")
    parser.add_argument("--dataset", type=str, default="daic_woz",
                        choices=["daic_woz", "modma", "pdch"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "dev", "val", "test"])
    parser.add_argument("--per_modality", action="store_true",
                        help="Evaluate all modality combinations")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_test_dataset(config: dict, dataset_name: str, split: str, data_root: str = None):
    ds_cfg = config.get("dataset", {})
    root = data_root or ds_cfg.get("root_dir", f"data/raw/{dataset_name}")
    phq8_thresh = ds_cfg.get("phq8_threshold", 10)
    modalities = ds_cfg.get("modalities", None)

    if dataset_name == "daic_woz":
        from data.datasets.daic_woz_dataset import DAICWOZDataset
        return DAICWOZDataset(root, split=split, phq8_threshold=phq8_thresh,
                               modalities=modalities)
    elif dataset_name == "modma":
        from data.datasets.modma_dataset import MODMADataset
        return MODMADataset(root, split=split, phq8_threshold=phq8_thresh,
                             modalities=modalities)
    elif dataset_name == "pdch":
        from data.datasets.pdch_dataset import PDCHDataset
        return PDCHDataset(root, split=split, phq8_threshold=phq8_thresh,
                            modalities=modalities)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)

    # Setup logger
    experiment_logger = Logger(
        name="dg-hmcf-eval",
        log_dir=os.path.join(args.output_dir, "logs"),
        use_wandb=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    experiment_logger.info("Device: %s", device)

    # ---- Load model ---------------------------------------------------------
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    state = torch.load(args.checkpoint, map_location=device)
    model_config = state.get("config", config).get("model", config.get("model", {}))
    model = DGHMCF(model_config)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()
    experiment_logger.info("Loaded checkpoint from epoch %d.", state.get("epoch", -1))

    # ---- Dataset ------------------------------------------------------------
    test_ds = build_test_dataset(config, args.dataset, args.split, args.data_root)
    experiment_logger.info("Test set size: %d", len(test_ds))

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ---- Evaluate -----------------------------------------------------------
    evaluator = Evaluator(model, config, device)
    metrics = evaluator.evaluate(test_loader)

    experiment_logger.info("=" * 60)
    experiment_logger.info("EVALUATION RESULTS")
    experiment_logger.info("=" * 60)
    for k, v in metrics.items():
        experiment_logger.info("  %-20s: %.4f", k, v)

    # ---- Per-modality evaluation --------------------------------------------
    if args.per_modality:
        experiment_logger.info("Running per-modality combination evaluation...")
        combo_results = evaluator.evaluate_per_modality_combination(test_loader)
        import pandas as pd
        combo_df = pd.DataFrame(
            [{
                "combination": k,
                **{metric: round(v, 4) for metric, v in vals.items()}
            }
            for k, vals in combo_results.items()]
        )
        combo_path = os.path.join(args.output_dir, "modality_combination_results.csv")
        combo_df.to_csv(combo_path, index=False)
        experiment_logger.info("Per-modality results saved to %s", combo_path)

    # ---- Generate predictions -----------------------------------------------
    predictions_df = evaluator.generate_predictions(test_loader)
    pred_path = os.path.join(args.output_dir, "predictions.csv")
    predictions_df.to_csv(pred_path, index=False)
    experiment_logger.info("Predictions saved to %s", pred_path)

    # ---- Visualisations -----------------------------------------------------
    import numpy as np

    viz_dir = os.path.join(args.output_dir, "figures")

    plot_confusion_matrix(
        labels=predictions_df["true_label"].values,
        preds=predictions_df["predicted_label"].values,
        save_path=os.path.join(viz_dir, "confusion_matrix.png"),
    )

    plot_phq8_scatter(
        true_scores=predictions_df["true_phq8_raw"].values,
        pred_scores=predictions_df["pred_phq8_raw"].values,
        save_path=os.path.join(viz_dir, "phq8_scatter.png"),
    )

    # Reliability weight analysis
    weight_df = evaluator.analyze_reliability_weights(test_loader)
    weight_cols = [c for c in weight_df.columns if c.startswith("weight_")]
    if weight_cols:
        weights_arr = weight_df[weight_cols].values
        plot_reliability_weights(
            weights_arr,
            modality_names=[c.replace("weight_", "") for c in weight_cols],
            save_path=os.path.join(viz_dir, "reliability_weights.png"),
        )

    experiment_logger.info("Evaluation complete. Figures saved to %s", viz_dir)
    experiment_logger.finish()


if __name__ == "__main__":
    main()
