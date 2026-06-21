"""
Evaluation metrics for DG-HMCF.

Provides:
  - compute_classification_metrics : accuracy, F1, precision, recall, AUROC
  - compute_regression_metrics     : MAE, RMSE, Pearson r
  - DepressionMetrics              : torchmetrics-based tracker for training loops
"""

from typing import Dict, Optional

import numpy as np
import torch

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        mean_absolute_error,
        mean_squared_error,
    )
    from scipy.stats import pearsonr
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from torchmetrics import (
        Accuracy,
        F1Score,
        Precision,
        Recall,
        AUROC,
        MeanAbsoluteError,
        MeanSquaredError,
        PearsonCorrCoef,
    )
    _TORCHMETRICS_AVAILABLE = True
except ImportError:
    _TORCHMETRICS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Parameters
    ----------
    preds  : np.ndarray of int, shape (N,)  – predicted class indices
    labels : np.ndarray of int, shape (N,)  – true class indices
    probs  : np.ndarray of float, shape (N,) or (N, 2) – class probabilities
             used for AUROC; if None, AUROC is omitted.

    Returns
    -------
    dict with keys: accuracy, f1, precision, recall, (auc)
    """
    if not _SKLEARN_AVAILABLE:
        return _compute_cls_fallback(preds, labels)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="binary", zero_division=0)),
        "precision": float(precision_score(labels, preds, average="binary", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="binary", zero_division=0)),
    }

    if probs is not None:
        try:
            prob_pos = probs[:, 1] if probs.ndim == 2 else probs
            metrics["auc"] = float(roc_auc_score(labels, prob_pos))
        except ValueError:
            metrics["auc"] = 0.0

    return metrics


def compute_regression_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    scale: float = 24.0,
) -> Dict[str, float]:
    """
    Compute regression metrics on PHQ-8 scores.

    Parameters
    ----------
    preds  : np.ndarray of float, shape (N,) – predicted normalised scores [0, 1]
    labels : np.ndarray of float, shape (N,) – true normalised scores [0, 1]
    scale  : float – multiply by this to convert to raw PHQ-8 (default 24)

    Returns
    -------
    dict with keys: mae, rmse, pearson_r (all in raw PHQ-8 scale)
    """
    preds_raw = preds * scale
    labels_raw = labels * scale

    if not _SKLEARN_AVAILABLE:
        return _compute_reg_fallback(preds_raw, labels_raw)

    mae = float(mean_absolute_error(labels_raw, preds_raw))
    rmse = float(np.sqrt(mean_squared_error(labels_raw, preds_raw)))
    try:
        r, _ = pearsonr(preds_raw, labels_raw)
        pearson = float(r) if not np.isnan(r) else 0.0
    except Exception:
        pearson = 0.0

    return {"mae": mae, "rmse": rmse, "pearson_r": pearson}


def _compute_cls_fallback(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Pure-numpy fallback for classification metrics."""
    correct = np.sum(preds == labels)
    n = len(labels)
    accuracy = float(correct / n) if n > 0 else 0.0

    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def _compute_reg_fallback(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Pure-numpy fallback for regression metrics."""
    mae = float(np.mean(np.abs(preds - labels)))
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
    # Pearson r without scipy
    if np.std(preds) < 1e-9 or np.std(labels) < 1e-9:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(preds, labels)[0, 1])
    return {"mae": mae, "rmse": rmse, "pearson_r": pearson}


# ---------------------------------------------------------------------------
# Stateful Metric Tracker
# ---------------------------------------------------------------------------

class DepressionMetrics:
    """
    Stateful metric tracker for use inside training / validation loops.

    Accumulates predictions and labels across batches, then computes
    all metrics at epoch end.

    Parameters
    ----------
    device : str or torch.device
        Device to keep tensors on.
    use_torchmetrics : bool
        Use torchmetrics if available (GPU-compatible).
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        use_torchmetrics: bool = True,
    ) -> None:
        self.device = device
        self.reset()

        self.use_torchmetrics = use_torchmetrics and _TORCHMETRICS_AVAILABLE
        if self.use_torchmetrics:
            self._init_torchmetrics()

    def _init_torchmetrics(self) -> None:
        """Initialise torchmetrics objects."""
        self.tm_accuracy = Accuracy(task="binary").to(self.device)
        self.tm_f1 = F1Score(task="binary").to(self.device)
        self.tm_precision = Precision(task="binary").to(self.device)
        self.tm_recall = Recall(task="binary").to(self.device)
        self.tm_auroc = AUROC(task="binary").to(self.device)
        self.tm_mae = MeanAbsoluteError().to(self.device)
        self.tm_rmse = MeanSquaredError(squared=False).to(self.device)
        self.tm_pearson = PearsonCorrCoef().to(self.device)

    def reset(self) -> None:
        """Clear accumulated predictions."""
        self._preds_cls: list = []
        self._probs_cls: list = []
        self._labels_cls: list = []
        self._preds_reg: list = []
        self._labels_reg: list = []

        if hasattr(self, "tm_accuracy") and self.use_torchmetrics:
            for attr in [
                "tm_accuracy", "tm_f1", "tm_precision", "tm_recall",
                "tm_auroc", "tm_mae", "tm_rmse", "tm_pearson"
            ]:
                getattr(self, attr).reset()

    def update(
        self,
        logits: torch.Tensor,
        phq8_pred: torch.Tensor,
        labels: torch.Tensor,
        phq8_labels: torch.Tensor,
    ) -> None:
        """
        Accumulate one batch of predictions.

        Parameters
        ----------
        logits      : (B, 2) classification logits
        phq8_pred   : (B,)  normalised PHQ-8 predictions
        labels      : (B,)  integer true labels
        phq8_labels : (B,)  normalised true PHQ-8 scores
        """
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        self._preds_cls.append(preds.detach().cpu())
        self._probs_cls.append(probs.detach().cpu())
        self._labels_cls.append(labels.detach().cpu())
        self._preds_reg.append(phq8_pred.detach().cpu())
        self._labels_reg.append(phq8_labels.detach().cpu())

        if self.use_torchmetrics:
            preds_d = preds.to(self.device)
            labels_d = labels.long().to(self.device)
            probs_pos = probs[:, 1].to(self.device)
            phq8_pred_d = phq8_pred.to(self.device)
            phq8_labels_d = phq8_labels.to(self.device)

            self.tm_accuracy.update(preds_d, labels_d)
            self.tm_f1.update(preds_d, labels_d)
            self.tm_precision.update(preds_d, labels_d)
            self.tm_recall.update(preds_d, labels_d)
            self.tm_auroc.update(probs_pos, labels_d)
            self.tm_mae.update(phq8_pred_d * 24.0, phq8_labels_d * 24.0)
            self.tm_rmse.update(phq8_pred_d * 24.0, phq8_labels_d * 24.0)
            self.tm_pearson.update(phq8_pred_d * 24.0, phq8_labels_d * 24.0)

    def compute(self) -> Dict[str, float]:
        """Compute and return all accumulated metrics."""
        if self.use_torchmetrics:
            try:
                return {
                    "accuracy": float(self.tm_accuracy.compute()),
                    "f1": float(self.tm_f1.compute()),
                    "precision": float(self.tm_precision.compute()),
                    "recall": float(self.tm_recall.compute()),
                    "auc": float(self.tm_auroc.compute()),
                    "mae": float(self.tm_mae.compute()),
                    "rmse": float(self.tm_rmse.compute()),
                    "pearson_r": float(self.tm_pearson.compute()),
                }
            except Exception:
                pass  # Fall through to numpy implementation

        # Fallback numpy implementation
        all_preds = torch.cat(self._preds_cls).numpy()
        all_probs = torch.cat(self._probs_cls).numpy()
        all_labels = torch.cat(self._labels_cls).numpy()
        all_reg_preds = torch.cat(self._preds_reg).numpy()
        all_reg_labels = torch.cat(self._labels_reg).numpy()

        cls_metrics = compute_classification_metrics(all_preds, all_labels, all_probs)
        reg_metrics = compute_regression_metrics(all_reg_preds, all_reg_labels)

        return {**cls_metrics, **reg_metrics}
