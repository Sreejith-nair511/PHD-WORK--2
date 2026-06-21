"""
Depression Classifier for DG-HMCF.

Multi-task head with:
  - Branch 1: binary classification (depressed / non-depressed)
  - Branch 2: PHQ-8 score regression (0–24)
"""

from typing import Tuple

import torch
import torch.nn as nn


class DepressionClassifier(nn.Module):
    """
    Multi-task output head.

    Takes a fused feature vector and produces:
      1. ``classification_logits`` – shape (B, n_classes), raw logits
      2. ``phq8_score``            – shape (B,), normalised score in [0, 1]
                                     (multiply by 24 to recover raw PHQ-8)

    Parameters
    ----------
    fusion_dim : int
        Dimensionality of the input fused representation.
    hidden_dim : int
        Hidden units in the classifier MLP.
    n_classes : int
        Number of output classes (default 2: non-depressed / depressed).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        fusion_dim: int = 512,
        hidden_dim: int = 256,
        n_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # ---- Shared trunk ------------------------------------------------
        self.shared = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- Branch 1: Classification ------------------------------------
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        # ---- Branch 2: PHQ-8 Regression ----------------------------------
        # Output: normalised PHQ-8 score in [0, 1]
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),   # constrain to [0, 1]
        )

    def forward(
        self, fused_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        fused_features : torch.Tensor, shape (B, fusion_dim)

        Returns
        -------
        classification_logits : torch.Tensor, shape (B, n_classes)
        phq8_score            : torch.Tensor, shape (B,)
            Normalised PHQ-8 prediction in [0, 1].
        """
        shared_feat = self.shared(fused_features)              # (B, hidden_dim)

        cls_logits = self.cls_head(shared_feat)                # (B, n_classes)
        phq8 = self.reg_head(shared_feat).squeeze(-1)          # (B,)

        return cls_logits, phq8
