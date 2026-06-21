"""
Dynamic Reliability Gating module for DG-HMCF.

Computes per-modality reliability weights based on the quality of each
embedding, masked by modality availability, and temperature-scaled softmax.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class DynamicReliabilityGating(nn.Module):
    """
    Assign data-driven reliability weights to present modalities.

    For each modality embedding, a small MLP scores its "quality" (e.g.,
    how informative it is for the current sample).  Absent modalities are
    masked to −∞ before the softmax so they receive zero weight.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of each modality embedding.
    n_modalities : int
        Number of modalities (default 4: speech, text, face, EEG).
    hidden_dim : int
        Hidden units in the quality-scoring MLP.
    temperature : float
        Softmax temperature τ.  Higher → softer weights; lower → sharper.
    """

    MODALITY_ORDER = ["speech", "text", "face", "eeg"]

    def __init__(
        self,
        embed_dim: int = 256,
        n_modalities: int = 4,
        hidden_dim: int = 128,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_modalities = n_modalities
        self.temperature = nn.Parameter(
            torch.tensor(temperature), requires_grad=True
        )

        # Independent quality-scoring MLP for each modality
        self.quality_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(n_modalities)
            ]
        )

        # Global context aggregator: attends to all modality embeddings
        # to produce a global context that modulates scores
        self.context_proj = nn.Linear(embed_dim, hidden_dim)
        self.score_modulate = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        embeddings: List[Optional[torch.Tensor]],
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : list of length n_modalities
            Each element is either a tensor of shape (B, embed_dim) or None.
            None values are replaced internally with zero vectors.
        modality_mask : torch.Tensor, shape (B, n_modalities)
            Binary flags: 1 if modality is present, 0 if absent.

        Returns
        -------
        reliability_weights : torch.Tensor, shape (B, n_modalities)
            Reliability weights summing to 1 over present modalities.
            Zero weight for absent modalities.
        """
        device = modality_mask.device
        B = modality_mask.size(0)

        # Replace None embeddings with zero vectors
        emb_list: List[torch.Tensor] = []
        for i, emb in enumerate(embeddings):
            if emb is None:
                # Create zero placeholder
                embed_dim = self.quality_mlps[i][0].in_features
                emb_list.append(torch.zeros(B, embed_dim, device=device))
            else:
                emb_list.append(emb)

        # Compute global context as mean of present embeddings
        present_mask = modality_mask.unsqueeze(-1)  # (B, n_modalities, 1)
        stacked_embs = torch.stack(emb_list, dim=1)  # (B, n_modalities, embed_dim)
        context = (stacked_embs * present_mask).sum(dim=1) / (
            present_mask.sum(dim=1).clamp(min=1.0)
        )  # (B, embed_dim)
        context_h = torch.relu(self.context_proj(context))  # (B, hidden_dim)

        # Score each modality
        scores = []
        for i, emb in enumerate(emb_list):
            # Individual quality score
            indiv_score = self.quality_mlps[i](emb)  # (B, 1)

            # Context-modulated score
            emb_h = torch.relu(
                nn.functional.linear(
                    emb,
                    self.quality_mlps[i][0].weight,
                    self.quality_mlps[i][0].bias,
                )
            )  # (B, hidden_dim)
            combined = torch.cat([emb_h, context_h], dim=-1)  # (B, 2*hidden_dim)
            mod_score = self.score_modulate(combined)  # (B, 1)

            scores.append(indiv_score + mod_score)  # (B, 1)

        scores_cat = torch.cat(scores, dim=-1)  # (B, n_modalities)

        # Mask absent modalities with a large negative value
        NEG_INF = -1e9
        mask_bool = modality_mask.bool()
        scores_masked = scores_cat.masked_fill(~mask_bool, NEG_INF)

        # Temperature-scaled softmax
        temp = self.temperature.abs().clamp(min=1e-3)
        weights = torch.softmax(scores_masked / temp, dim=-1)  # (B, n_modalities)

        # Zero out absent modalities (softmax can still give small values
        # when all logits are -inf → NaN; handle that edge case)
        weights = weights * modality_mask.float()
        # Re-normalise in case of floating point residuals
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        weights = weights / weight_sum

        return weights
