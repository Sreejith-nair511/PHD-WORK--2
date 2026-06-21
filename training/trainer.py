"""
Trainer for DG-HMCF.

Implements the full training loop with:
  - AdamW optimiser with linear warmup + cosine schedule
  - Gradient clipping
  - Early stopping
  - Checkpoint save/load
  - Optional WandB logging
"""

import os
import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.losses import MultiTaskDepressionLoss
from training.metrics import DepressionMetrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training orchestrator for the DG-HMCF model.

    Parameters
    ----------
    model        : nn.Module  – the DG-HMCF model
    config       : dict       – full training configuration
    train_loader : DataLoader – training data loader
    val_loader   : DataLoader – validation data loader
    device       : torch.device
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        tr_cfg = config.get("training", config)

        # ---- Criterion ---------------------------------------------------
        loss_cfg = tr_cfg.get("loss", {})
        self.criterion = MultiTaskDepressionLoss(
            cls_weight=loss_cfg.get("classification_weight", 0.5),
            reg_weight=loss_cfg.get("regression_weight", 0.5),
            label_smoothing=0.1,
        )

        # ---- Optimiser ---------------------------------------------------
        self.lr = float(tr_cfg.get("learning_rate", 1e-4))
        self.weight_decay = float(tr_cfg.get("weight_decay", 1e-5))
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # ---- Scheduler ---------------------------------------------------
        self.epochs = int(tr_cfg.get("epochs", 50))
        warmup_steps = int(tr_cfg.get("warmup_steps", 500))
        total_steps = self.epochs * max(len(train_loader), 1)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=self.lr * 0.01,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        # ---- Gradient clipping -------------------------------------------
        self.gradient_clip = float(tr_cfg.get("gradient_clip", 1.0))

        # ---- Early stopping ----------------------------------------------
        self.patience = int(tr_cfg.get("early_stopping_patience", 10))
        self.best_val_f1 = -np.inf
        self.early_stop_counter = 0

        # ---- Logging -----------------------------------------------------
        log_cfg = config.get("logging", {})
        self.checkpoint_dir = log_cfg.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_interval = int(log_cfg.get("log_interval", 10))

        self.use_wandb = log_cfg.get("use_wandb", False)
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=log_cfg.get("wandb_project", "dg-hmcf"),
                    config=config,
                )
                self.wandb = wandb
            except Exception:
                self.use_wandb = False

        # Training history
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "val_accuracy": [], "val_f1": [], "val_auc": [],
            "val_mae": [], "val_rmse": [],
        }

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop.

        Returns
        -------
        dict with history and best validation metrics.
        """
        logger.info("Starting training for %d epochs.", self.epochs)
        best_metrics: Dict[str, float] = {}

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            # ---- Train one epoch ----------------------------------------
            train_metrics = self.train_epoch()
            self.history["train_loss"].append(train_metrics["total_loss"])

            # ---- Validate -----------------------------------------------
            val_metrics = self.validate()
            for key in ("loss", "accuracy", "f1", "auc", "mae", "rmse"):
                hist_key = f"val_{key}"
                if hist_key in self.history:
                    self.history[hist_key].append(val_metrics.get(key, 0.0))

            elapsed = time.time() - t0
            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
                "val_f1=%.4f | val_auc=%.4f | val_mae=%.3f | %.1fs",
                epoch, self.epochs,
                train_metrics["total_loss"],
                val_metrics.get("loss", 0.0),
                val_metrics.get("f1", 0.0),
                val_metrics.get("auc", 0.0),
                val_metrics.get("mae", 0.0),
                elapsed,
            )

            if self.use_wandb:
                self.wandb.log({
                    "epoch": epoch,
                    "train/loss": train_metrics["total_loss"],
                    "val/loss": val_metrics.get("loss", 0.0),
                    "val/f1": val_metrics.get("f1", 0.0),
                    "val/auc": val_metrics.get("auc", 0.0),
                    "val/mae": val_metrics.get("mae", 0.0),
                })

            # ---- Early stopping check -----------------------------------
            current_f1 = val_metrics.get("f1", 0.0)
            if current_f1 > self.best_val_f1:
                self.best_val_f1 = current_f1
                best_metrics = val_metrics
                self.early_stop_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    logger.info(
                        "Early stopping triggered after %d epochs without improvement.",
                        self.patience,
                    )
                    break

            # ---- Periodic checkpoint ------------------------------------
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)

        if self.use_wandb:
            self.wandb.finish()

        return {"history": self.history, "best_metrics": best_metrics}

    # ------------------------------------------------------------------
    # Epoch-level loops
    # ------------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch, return average losses."""
        self.model.train()
        total_loss_sum = cls_loss_sum = reg_loss_sum = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for batch_idx, batch in enumerate(pbar):
            batch = self._to_device(batch)

            self.optimizer.zero_grad()

            outputs = self.model(batch)
            loss_dict = self.criterion(
                classification_logits=outputs["classification_logits"],
                phq8_pred=outputs["phq8_score"],
                classification_labels=batch["label"],
                phq8_labels=batch["phq8_score"],
            )

            loss = loss_dict["total_loss"]
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            )

            self.optimizer.step()
            self.scheduler.step()

            total_loss_sum += loss.item()
            cls_loss_sum += loss_dict["cls_loss"].item()
            reg_loss_sum += loss_dict["reg_loss"].item()
            n_batches += 1

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        n_batches = max(n_batches, 1)
        return {
            "total_loss": total_loss_sum / n_batches,
            "cls_loss": cls_loss_sum / n_batches,
            "reg_loss": reg_loss_sum / n_batches,
        }

    def validate(self) -> Dict[str, float]:
        """Run validation, return metrics dict."""
        self.model.eval()
        metrics_tracker = DepressionMetrics(device=self.device)
        loss_sum = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Val", leave=False):
                batch = self._to_device(batch)

                outputs = self.model(batch)
                loss_dict = self.criterion(
                    classification_logits=outputs["classification_logits"],
                    phq8_pred=outputs["phq8_score"],
                    classification_labels=batch["label"],
                    phq8_labels=batch["phq8_score"],
                )

                loss_sum += loss_dict["total_loss"].item()
                n_batches += 1

                metrics_tracker.update(
                    logits=outputs["classification_logits"],
                    phq8_pred=outputs["phq8_score"],
                    labels=batch["label"],
                    phq8_labels=batch["phq8_score"],
                )

        metrics = metrics_tracker.compute()
        metrics["loss"] = loss_sum / max(n_batches, 1)
        return metrics

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint to disk."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch:03d}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, path)
        logger.info("Saved checkpoint: %s", path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load a checkpoint from disk and restore model/optimiser state.

        Returns
        -------
        dict with ``epoch`` and ``metrics`` keys.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        logger.info(
            "Loaded checkpoint from epoch %d (metrics: %s)",
            state["epoch"],
            state.get("metrics", {}),
        )
        return {"epoch": state["epoch"], "metrics": state.get("metrics", {})}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively move all tensors in a batch dict to the target device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, dict):
                result[key] = self._to_device(value)
            else:
                result[key] = value
        return result
