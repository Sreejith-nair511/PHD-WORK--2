"""
Explainability Module for DG-HMCF.

Aggregates modality importance scores, attention maps, and reliability
weights into a human-readable and visualisation-friendly output dict.
"""

from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn


MODALITY_NAMES = ["speech", "text", "face", "eeg"]
PHQ8_SEVERITY = [
    (0, 4, "None"),
    (5, 9, "Mild"),
    (10, 14, "Moderate"),
    (15, 19, "Moderately Severe"),
    (20, 24, "Severe"),
]


class ExplainabilityModule(nn.Module):
    """
    Compute and format explanations for model predictions.

    Aggregates:
      - Modality reliability weights → importance scores
      - Cross-modal attention weights → attention heat maps
      - Per-modality embedding norms → feature salience proxy

    Parameters
    ----------
    embed_dim : int
        Dimensionality of modality embeddings.
    fusion_dim : int
        Dimensionality of the fused representation.
    n_modalities : int
        Number of modalities.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        fusion_dim: int = 512,
        n_modalities: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_modalities = n_modalities

        # Learnable scaling for combining reliability + norm-based importance
        self.importance_blend = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        reliability_weights: torch.Tensor,
        cross_modal_attention_weights: Dict[str, torch.Tensor],
        embeddings: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        reliability_weights : torch.Tensor, shape (B, n_modalities)
            Output of DynamicReliabilityGating.
        cross_modal_attention_weights : dict
            Keys like ``"speech_text_a2b"``, values are attention tensors
            (B, T_q, T_kv) or (B, 1, 1).
        embeddings : dict
            Modality embeddings, shape (B, embed_dim) per key.

        Returns
        -------
        dict with keys:
            ``modality_importance`` – (B, n_modalities) float tensor
            ``attention_maps``      – raw attention weight dict
            ``embedding_norms``     – (B, n_modalities) L2 norms
            ``reliability_weights`` – passthrough of input weights
        """
        B = reliability_weights.size(0)
        device = reliability_weights.device

        # ---- Embedding L2 norms -----------------------------------------
        norms = torch.zeros(B, self.n_modalities, device=device)
        for i, name in enumerate(MODALITY_NAMES):
            emb = embeddings.get(name)
            if emb is not None:
                norms[:, i] = emb.norm(p=2, dim=-1)

        # Normalise norms to [0, 1] per sample
        norm_sum = norms.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        norm_importance = norms / norm_sum  # (B, n_modalities)

        # ---- Blend reliability weights + norm importance -----------------
        alpha = self.importance_blend.sigmoid()  # in [0, 1]
        modality_importance = (
            alpha * reliability_weights + (1 - alpha) * norm_importance
        )

        return {
            "modality_importance": modality_importance,        # (B, 4)
            "attention_maps": cross_modal_attention_weights,   # dict
            "embedding_norms": norms,                          # (B, 4)
            "reliability_weights": reliability_weights,        # (B, 4)
        }

    def generate_report(
        self,
        explainability_output: Dict[str, Any],
        prediction: Dict[str, torch.Tensor],
        sample_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Produce a human-readable explanation for a single sample.

        Parameters
        ----------
        explainability_output : dict
            Output of ``forward()``.
        prediction : dict
            Model output containing ``classification_logits`` and ``phq8_score``.
        sample_idx : int
            Index within the batch to explain.

        Returns
        -------
        dict with narrative fields ready for display or logging.
        """
        modality_importance = (
            explainability_output["modality_importance"][sample_idx]
            .detach()
            .cpu()
            .numpy()
        )
        reliability = (
            explainability_output["reliability_weights"][sample_idx]
            .detach()
            .cpu()
            .numpy()
        )
        norms = (
            explainability_output["embedding_norms"][sample_idx]
            .detach()
            .cpu()
            .numpy()
        )

        # Classification result
        logits = prediction["classification_logits"][sample_idx].detach().cpu()
        probs = torch.softmax(logits, dim=-1).numpy()
        pred_label = int(logits.argmax().item())

        # PHQ-8 score
        phq8_norm = float(prediction["phq8_score"][sample_idx].detach().cpu().item())
        phq8_raw = phq8_norm * 24.0  # denormalize

        severity = "Unknown"
        for lo, hi, label in PHQ8_SEVERITY:
            if lo <= phq8_raw <= hi:
                severity = label
                break

        # Modality importance ranking
        importance_ranking = sorted(
            zip(MODALITY_NAMES, modality_importance.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        report = {
            "prediction": "Depressed" if pred_label == 1 else "Non-Depressed",
            "confidence": float(probs[pred_label]),
            "phq8_score_estimated": round(phq8_raw, 1),
            "severity": severity,
            "modality_importance_ranking": [
                {"modality": m, "importance": round(imp, 4)}
                for m, imp in importance_ranking
            ],
            "reliability_weights": {
                name: round(float(reliability[i]), 4)
                for i, name in enumerate(MODALITY_NAMES)
            },
            "embedding_norms": {
                name: round(float(norms[i]), 4)
                for i, name in enumerate(MODALITY_NAMES)
            },
            "summary": (
                f"The model predicts '{('Depressed' if pred_label == 1 else 'Non-Depressed')}' "
                f"(confidence: {probs[pred_label]:.1%}) with estimated PHQ-8 score "
                f"{phq8_raw:.1f} ({severity} depression). "
                f"Most influential modality: {importance_ranking[0][0]}."
            ),
        }
        return report
