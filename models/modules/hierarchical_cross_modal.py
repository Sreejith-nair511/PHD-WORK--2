"""
Hierarchical Cross-Modal Transformer module for DG-HMCF.

Performs pairwise cross-attention between all present modality pairs
in a fixed hierarchical order, then returns enhanced per-modality embeddings.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttentionBlock(nn.Module):
    """
    Single cross-attention block: query attends to key/value from another modality.

    Architecture:
      MultiHead-CrossAttention → Residual → LayerNorm
      FFN → Residual → LayerNorm

    Parameters
    ----------
    embed_dim : int
        Embedding dimension of both query and key/value.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    ffn_dim : int
        Hidden dimension of the feed-forward network.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        ffn_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Cross-attention: query from modality A, key/value from modality B
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query : torch.Tensor
            Shape (B, 1, embed_dim) or (B, T_q, embed_dim).
        key_value : torch.Tensor
            Shape (B, 1, embed_dim) or (B, T_kv, embed_dim).
        key_padding_mask : torch.Tensor, optional
            Shape (B, T_kv), True for positions to ignore.

        Returns
        -------
        enhanced_query : torch.Tensor, same shape as ``query``
        attn_weights   : torch.Tensor, shape (B, T_q, T_kv)
        """
        # Cross-attention
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        # Residual + LayerNorm
        query = self.norm1(query + self.dropout(attn_out))

        # Feed-forward
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)

        return query, attn_weights


class HierarchicalCrossModalTransformer(nn.Module):
    """
    Hierarchical cross-modal transformer with 6 pairwise attention stages.

    Cross-attention pairs (in order):
      1. speech ↔ text
      2. speech ↔ face
      3. speech ↔ eeg
      4. text   ↔ face
      5. text   ↔ eeg
      6. face   ↔ eeg

    Each pair runs two cross-attention blocks (A→B and B→A) stacked
    ``n_layers`` times.  If either modality is absent (mask = 0), the
    pair is skipped and the embeddings pass through unchanged.

    Parameters
    ----------
    embed_dim : int
        Common embedding dimension for all modalities.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of cross-attention layers per pair.
    dropout : float
        Dropout probability.
    ffn_dim : int
        FFN hidden dimension.
    """

    # Modality index mapping
    MOD_IDX = {"speech": 0, "text": 1, "face": 2, "eeg": 3}
    PAIRS = [
        ("speech", "text"),
        ("speech", "face"),
        ("speech", "eeg"),
        ("text", "face"),
        ("text", "eeg"),
        ("face", "eeg"),
    ]

    def __init__(
        self,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        ffn_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        # For each pair and each direction, create n_layers cross-attn blocks
        # Organised as: pair_name → direction → list of blocks
        self.cross_attn_blocks: nn.ModuleDict = nn.ModuleDict()

        for mod_a, mod_b in self.PAIRS:
            pair_key = f"{mod_a}_{mod_b}"
            self.cross_attn_blocks[f"{pair_key}_a2b"] = nn.ModuleList(
                [
                    CrossModalAttentionBlock(embed_dim, n_heads, dropout, ffn_dim)
                    for _ in range(n_layers)
                ]
            )
            self.cross_attn_blocks[f"{pair_key}_b2a"] = nn.ModuleList(
                [
                    CrossModalAttentionBlock(embed_dim, n_heads, dropout, ffn_dim)
                    for _ in range(n_layers)
                ]
            )

    def forward(
        self,
        speech_emb: Optional[torch.Tensor],
        text_emb: Optional[torch.Tensor],
        face_emb: Optional[torch.Tensor],
        eeg_emb: Optional[torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        speech_emb : torch.Tensor or None, shape (B, embed_dim)
        text_emb   : torch.Tensor or None, shape (B, embed_dim)
        face_emb   : torch.Tensor or None, shape (B, embed_dim)
        eeg_emb    : torch.Tensor or None, shape (B, embed_dim)
        modality_mask : torch.Tensor, shape (B, 4)
            Column order: [speech, text, face, eeg].

        Returns
        -------
        dict with keys 'speech', 'text', 'face', 'eeg'
            Enhanced embeddings (same shape as inputs), or None for absent modalities.
        attention_maps : dict of pair → (B, 1, 1) attention weight tensors
        """
        # Build mutable working copies
        embs: Dict[str, Optional[torch.Tensor]] = {
            "speech": speech_emb,
            "text": text_emb,
            "face": face_emb,
            "eeg": eeg_emb,
        }
        attention_maps: Dict[str, torch.Tensor] = {}

        # Per-sample availability: (B,) bool tensor for each modality
        mask_dict = {
            "speech": modality_mask[:, 0],
            "text": modality_mask[:, 1],
            "face": modality_mask[:, 2],
            "eeg": modality_mask[:, 3],
        }

        for mod_a, mod_b in self.PAIRS:
            # Skip pair if either modality is absent in *all* samples
            if embs[mod_a] is None or embs[mod_b] is None:
                continue

            # Check if pair is valid for at least some samples
            both_present = mask_dict[mod_a] * mask_dict[mod_b]  # (B,)
            if both_present.sum() == 0:
                continue

            pair_key = f"{mod_a}_{mod_b}"
            emb_a = embs[mod_a]  # (B, embed_dim)
            emb_b = embs[mod_b]

            # Add sequence dimension for MultiheadAttention: (B, 1, embed_dim)
            q_a = emb_a.unsqueeze(1)
            q_b = emb_b.unsqueeze(1)

            # A attends to B (n_layers)
            last_attn_a2b = None
            for layer in self.cross_attn_blocks[f"{pair_key}_a2b"]:
                q_a, last_attn_a2b = layer(q_a, q_b)

            # B attends to A (n_layers)
            last_attn_b2a = None
            for layer in self.cross_attn_blocks[f"{pair_key}_b2a"]:
                q_b, last_attn_b2a = layer(q_b, q_a)

            # Remove sequence dimension
            enhanced_a = q_a.squeeze(1)  # (B, embed_dim)
            enhanced_b = q_b.squeeze(1)

            # Only update embeddings for samples where both modalities are present
            mask_expand = both_present.unsqueeze(-1).float()  # (B, 1)
            embs[mod_a] = enhanced_a * mask_expand + emb_a * (1 - mask_expand)
            embs[mod_b] = enhanced_b * mask_expand + emb_b * (1 - mask_expand)

            if last_attn_a2b is not None:
                attention_maps[f"{pair_key}_a2b"] = last_attn_a2b
            if last_attn_b2a is not None:
                attention_maps[f"{pair_key}_b2a"] = last_attn_b2a

        return embs, attention_maps
