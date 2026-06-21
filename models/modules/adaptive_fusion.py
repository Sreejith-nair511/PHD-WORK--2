"""
Adaptive Fusion Layer for DG-HMCF.

Combines per-modality embeddings using:
  1. Reliability-weighted sum of original embeddings
  2. Sum of cross-modal enhanced embeddings
  3. Residual path from individual embeddings
  4. MLP projection to fusion_dim
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class AdaptiveFusionLayer(nn.Module):
    """
    Fuse multi-modal embeddings into a single representation.

    Fusion formula:
        fused = alpha * weighted_sum(embs) + beta * cross_modal_sum + residual

    where alpha, beta are learnable scalars and weighted_sum uses the
    dynamic reliability weights from the gating module.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of each modality embedding.
    fusion_dim : int
        Dimensionality of the output fused representation.
    n_modalities : int
        Number of modalities.
    dropout : float
        Dropout probability.
    """

    MODALITY_ORDER = ["speech", "text", "face", "eeg"]

    def __init__(
        self,
        embed_dim: int = 256,
        fusion_dim: int = 512,
        n_modalities: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_dim = fusion_dim
        self.n_modalities = n_modalities

        # Learnable fusion scalars
        self.alpha = nn.Parameter(torch.ones(1))   # weight for reliability-weighted sum
        self.beta = nn.Parameter(torch.ones(1))    # weight for cross-modal sum
        self.gamma = nn.Parameter(torch.ones(1))   # weight for residual path

        # Per-modality input layer norms
        self.input_norms = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(n_modalities)]
        )

        # Cross-modal sum layer norm
        self.cross_norm = nn.LayerNorm(embed_dim)

        # Fusion MLP: maps concatenated evidence to fusion_dim
        # Input: 3 * embed_dim (reliability-weighted, cross-modal, residual)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
        )

        # Gating mechanism: decide how much of each stream to use
        self.stream_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, 3),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        embeddings: Dict[str, Optional[torch.Tensor]],
        reliability_weights: torch.Tensor,
        cross_modal_embeddings: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : dict
            Keys: 'speech', 'text', 'face', 'eeg'.
            Values: tensors of shape (B, embed_dim) or None.
        reliability_weights : torch.Tensor, shape (B, n_modalities)
            Weights from the DynamicReliabilityGating module.
        cross_modal_embeddings : dict
            Same structure as ``embeddings`` but enhanced by cross-modal attention.

        Returns
        -------
        torch.Tensor, shape (B, fusion_dim)
        """
        B = reliability_weights.size(0)
        device = reliability_weights.device

        # ---- Stream 1: reliability-weighted sum of original embeddings ----
        weighted_sum = torch.zeros(B, self.embed_dim, device=device)
        for i, name in enumerate(self.MODALITY_ORDER):
            emb = embeddings.get(name)
            if emb is not None:
                norm_emb = self.input_norms[i](emb)
                w = reliability_weights[:, i].unsqueeze(-1)  # (B, 1)
                weighted_sum = weighted_sum + w * norm_emb

        # ---- Stream 2: mean of cross-modal enhanced embeddings -----------
        cross_sum = torch.zeros(B, self.embed_dim, device=device)
        n_cross = 0
        for i, name in enumerate(self.MODALITY_ORDER):
            emb_cross = cross_modal_embeddings.get(name)
            if emb_cross is not None:
                cross_sum = cross_sum + emb_cross
                n_cross += 1
        if n_cross > 0:
            cross_sum = cross_sum / n_cross
        cross_sum = self.cross_norm(cross_sum)

        # ---- Stream 3: residual (simple sum of present original embeddings)
        residual = torch.zeros(B, self.embed_dim, device=device)
        n_present = 0
        for name in self.MODALITY_ORDER:
            emb = embeddings.get(name)
            if emb is not None:
                residual = residual + emb
                n_present += 1
        if n_present > 0:
            residual = residual / n_present

        # ---- Adaptive gating across streams ------------------------------
        concat_streams = torch.cat([weighted_sum, cross_sum, residual], dim=-1)  # (B, 3*embed)
        gate = self.stream_gate(concat_streams)  # (B, 3)

        # Apply gates
        gated = (
            gate[:, 0:1] * weighted_sum
            + gate[:, 1:2] * cross_sum
            + gate[:, 2:3] * residual
        )

        # ---- Fusion MLP --------------------------------------------------
        fused_concat = torch.cat([weighted_sum, cross_sum, gated], dim=-1)  # (B, 3*embed)
        output = self.fusion_mlp(fused_concat)  # (B, fusion_dim)

        return output
