"""
Evaluator for DG-HMCF.

Provides:
  - Full test-set evaluation
  - Per-modality-combination evaluation
  - Prediction DataFrame export
"""

import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.metrics import (
    DepressionMetrics,
    compute_classification_metrics,
    compute_regression_metrics,
)

logger = logging.getLogger(__name__)

MODALITY_NAMES = ["speech", "text", "face", "eeg"]


class Evaluator:
    """
    Comprehensive evaluation pipeline for the DG-HMCF model.

    Parameters
    ----------
    model  : nn.Module       – trained DG-HMCF model
    config : dict            – experiment configuration
    device : torch.device
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.config = config

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Compute all metrics on the full test set.

        Parameters
        ----------
        test_loader : DataLoader

        Returns
        -------
        dict with classification + regression metrics.
        """
        self.model.eval()
        metrics_tracker = DepressionMetrics(device=self.device)

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = self._to_device(batch)
                outputs = self.model(batch)
                metrics_tracker.update(
                    logits=outputs["classification_logits"],
                    phq8_pred=outputs["phq8_score"],
                    labels=batch["label"],
                    phq8_labels=batch["phq8_score"],
                )

        metrics = metrics_tracker.compute()
        logger.info("Test metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Per-modality-combination evaluation
    # ------------------------------------------------------------------

    def evaluate_per_modality_combination(
        self, test_loader: DataLoader
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance for all 15 non-empty subsets of 4 modalities.

        For each combination, the modality_mask is overridden to simulate
        only those modalities being available.

        Returns
        -------
        dict mapping combo_name → metrics_dict
        """
        self.model.eval()
        results: Dict[str, Dict[str, float]] = {}

        # Generate all non-empty subsets
        for r in range(1, len(MODALITY_NAMES) + 1):
            for combo in itertools.combinations(range(len(MODALITY_NAMES)), r):
                combo_name = "+".join(MODALITY_NAMES[i] for i in combo)
                combo_mask = torch.zeros(len(MODALITY_NAMES))
                for i in combo:
                    combo_mask[i] = 1.0

                metrics_tracker = DepressionMetrics(device=self.device)

                with torch.no_grad():
                    for batch in test_loader:
                        batch = self._to_device(batch)

                        # Override modality mask
                        B = batch["modality_mask"].size(0)
                        forced_mask = combo_mask.unsqueeze(0).expand(B, -1).to(self.device)
                        # Only keep modalities that were actually present
                        batch["modality_mask"] = batch["modality_mask"] * forced_mask

                        outputs = self.model(batch)
                        metrics_tracker.update(
                            logits=outputs["classification_logits"],
                            phq8_pred=outputs["phq8_score"],
                            labels=batch["label"],
                            phq8_labels=batch["phq8_score"],
                        )

                metrics = metrics_tracker.compute()
                results[combo_name] = metrics
                logger.info("Modality combo %s: f1=%.4f auc=%.4f mae=%.3f",
                            combo_name,
                            metrics.get("f1", 0.0),
                            metrics.get("auc", 0.0),
                            metrics.get("mae", 0.0))

        return results

    # ------------------------------------------------------------------
    # Prediction export
    # ------------------------------------------------------------------

    def generate_predictions(self, test_loader: DataLoader) -> pd.DataFrame:
        """
        Run inference on the test set and return predictions as a DataFrame.

        Returns
        -------
        pd.DataFrame with columns:
            subject_id, true_label, predicted_label, prob_depressed,
            true_phq8_raw, pred_phq8_raw, correct
        """
        self.model.eval()
        rows: List[Dict] = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating predictions"):
                batch = self._to_device(batch)
                outputs = self.model(batch)

                probs = torch.softmax(outputs["classification_logits"], dim=-1)
                preds = probs.argmax(dim=-1)

                for i in range(len(preds)):
                    rows.append({
                        "subject_id": batch["subject_id"][i] if "subject_id" in batch else "",
                        "true_label": int(batch["label"][i].item()),
                        "predicted_label": int(preds[i].item()),
                        "prob_non_depressed": float(probs[i, 0].item()),
                        "prob_depressed": float(probs[i, 1].item()),
                        "true_phq8_norm": float(batch["phq8_score"][i].item()),
                        "pred_phq8_norm": float(outputs["phq8_score"][i].item()),
                        "true_phq8_raw": float(batch["phq8_score"][i].item() * 24),
                        "pred_phq8_raw": float(outputs["phq8_score"][i].item() * 24),
                        "correct": int(preds[i].item() == int(batch["label"][i].item())),
                    })

        df = pd.DataFrame(rows)
        return df

    # ------------------------------------------------------------------
    # Reliability weight analysis
    # ------------------------------------------------------------------

    def analyze_reliability_weights(
        self, test_loader: DataLoader
    ) -> pd.DataFrame:
        """
        Collect and analyse the dynamic reliability weights across the test set.

        Returns a DataFrame with per-subject modality reliability scores.
        """
        self.model.eval()
        rows: List[Dict] = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Reliability analysis"):
                batch = self._to_device(batch)
                outputs = self.model(batch)

                weights = outputs["reliability_weights"].cpu().numpy()
                labels = batch["label"].cpu().numpy()
                subject_ids = batch.get("subject_id", [""] * weights.shape[0])

                for i in range(weights.shape[0]):
                    rows.append({
                        "subject_id": subject_ids[i] if isinstance(subject_ids, list) else str(subject_ids[i]),
                        "label": int(labels[i]),
                        **{f"weight_{name}": float(weights[i, j])
                           for j, name in enumerate(MODALITY_NAMES)},
                    })

        df = pd.DataFrame(rows)
        return df

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, dict):
                result[key] = self._to_device(value)
            else:
                result[key] = value
        return result
