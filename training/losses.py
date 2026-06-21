"""
Multi-Task Loss Functions for DG-HMCF.

Combines:
  - Cross-entropy (with optional label smoothing) for binary classification
  - Smooth-L1 (Huber) loss for PHQ-8 score regression
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskDepressionLoss(nn.Module):
    """
    Combined multi-task loss for depression detection.

    L_total = cls_weight * L_cls + reg_weight * L_reg

    where:
      L_cls = cross-entropy with optional label smoothing
      L_reg = smooth-L1 (Huber) loss on normalised PHQ-8 scores

    Parameters
    ----------
    cls_weight : float
        Weight for the classification loss term.
    reg_weight : float
        Weight for the regression loss term.
    label_smoothing : float
        Label smoothing coefficient for cross-entropy (0 = no smoothing).
    class_weights : torch.Tensor, optional
        Class weights tensor of shape (n_classes,) for handling imbalance.
    huber_delta : float
        Delta parameter for the Smooth-L1/Huber loss.
    """

    def __init__(
        self,
        cls_weight: float = 0.5,
        reg_weight: float = 0.5,
        label_smoothing: float = 0.1,
        class_weights: torch.Tensor = None,
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.label_smoothing = label_smoothing
        self.huber_delta = huber_delta

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        classification_logits: torch.Tensor,
        phq8_pred: torch.Tensor,
        classification_labels: torch.Tensor,
        phq8_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        classification_logits : torch.Tensor, shape (B, n_classes)
        phq8_pred             : torch.Tensor, shape (B,)  – normalised [0, 1]
        classification_labels : torch.Tensor, shape (B,)  – int labels {0, 1}
        phq8_labels           : torch.Tensor, shape (B,)  – normalised PHQ-8 [0, 1]

        Returns
        -------
        dict with:
            ``total_loss`` – scalar tensor
            ``cls_loss``   – scalar tensor
            ``reg_loss``   – scalar tensor
        """
        # ---- Classification loss -----------------------------------------
        cls_labels = classification_labels.long()

        if self.class_weights is not None:
            class_weights = self.class_weights.to(classification_logits.device)
        else:
            class_weights = None

        cls_loss = F.cross_entropy(
            classification_logits,
            cls_labels,
            weight=class_weights,
            label_smoothing=self.label_smoothing,
        )

        # ---- Regression loss ---------------------------------------------
        # Huber / Smooth-L1 is more robust to outliers than MSE
        reg_loss = F.huber_loss(
            phq8_pred,
            phq8_labels.float(),
            delta=self.huber_delta,
        )

        # ---- Combined loss -----------------------------------------------
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for highly imbalanced depression datasets.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Can be used as a drop-in replacement for the classification branch of
    MultiTaskDepressionLoss.

    Parameters
    ----------
    alpha : float or list
        Class weight(s). Scalar for uniform weighting; list for per-class.
    gamma : float
        Focusing parameter (default 2.0).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (B, C) raw class logits
        labels : (B,)  integer class labels

        Returns
        -------
        Scalar focal loss.
        """
        B, C = logits.shape
        ce_loss = F.cross_entropy(logits, labels.long(), reduction="none")  # (B,)
        pt = torch.exp(-ce_loss)  # probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
