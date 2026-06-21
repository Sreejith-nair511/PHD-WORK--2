"""
Face Branch for DG-HMCF.

Combines a Vision Transformer (ViT) backbone with per-frame temporal
aggregation and behavioural feature injection.
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from transformers import ViTModel
except ImportError:
    ViTModel = None  # type: ignore


class FaceBranch(nn.Module):
    """
    Facial expression / action unit encoder branch.

    Architecture:
      1. ViT backbone over each frame   → (B, F, 768)  (F = n_frames)
      2. Temporal mean-pool             → (B, 768)
      3. Linear projection              → (B, embed_dim)
      4. Behavioural feature MLP        → (B, embed_dim)
      5. Fusion (add + LayerNorm)       → (B, embed_dim)

    Parameters
    ----------
    vit_model : str
        HuggingFace ViT model identifier.
    behavioral_feat_dim : int
        Size of the behavioural feature vector (default 7).
    embed_dim : int
        Output embedding dimension.
    dropout : float
        Dropout probability.
    freeze_vit : bool
        Freeze ViT parameters (useful for few-shot / limited data regimes).
    max_frames : int
        Maximum number of frames per sample (for efficient batching).
    """

    def __init__(
        self,
        vit_model: str = "google/vit-base-patch16-224",
        behavioral_feat_dim: int = 7,
        embed_dim: int = 256,
        dropout: float = 0.1,
        freeze_vit: bool = False,
        max_frames: int = 300,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.vit_hidden = 768
        self.max_frames = max_frames

        # ---- ViT backbone ------------------------------------------------
        if ViTModel is not None:
            try:
                self.vit = ViTModel.from_pretrained(vit_model)
                if freeze_vit:
                    for param in self.vit.parameters():
                        param.requires_grad = False
            except Exception:
                self.vit = None
        else:
            self.vit = None

        # ---- Projection from ViT to embed_dim ----------------------------
        self.visual_proj = nn.Sequential(
            nn.Linear(self.vit_hidden, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- Behavioural feature MLP -------------------------------------
        self.behavioral_proj = nn.Sequential(
            nn.Linear(behavioral_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- Temporal attention pooling (learned) ------------------------
        self.temporal_attn = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1),
        )

        # ---- Fusion layer ------------------------------------------------
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # ---- Output projection -------------------------------------------
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
        behavioral_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pixel_values : torch.Tensor, shape (B, F, C, H, W)
            Pre-normalised frame tensors.
        frame_mask : torch.Tensor, shape (B, F), optional
            1 for valid frames, 0 for padding.
        behavioral_features : torch.Tensor, shape (B, behavioral_feat_dim), optional

        Returns
        -------
        torch.Tensor, shape (B, embed_dim)
        """
        B, F, C, H, W = pixel_values.shape

        # ---- Per-frame ViT encoding --------------------------------------
        if self.vit is not None:
            # Process all frames in a single batch: (B*F, C, H, W)
            frames_flat = pixel_values.view(B * F, C, H, W)
            vit_out = self.vit(pixel_values=frames_flat)
            # Use [CLS] token output
            frame_features = vit_out.last_hidden_state[:, 0, :]  # (B*F, 768)
            frame_features = frame_features.view(B, F, self.vit_hidden)
        else:
            # Fallback: average-pool over spatial dims
            frames_flat = pixel_values.view(B * F, C, H, W)
            frame_features = frames_flat.mean(dim=[2, 3])  # (B*F, C)
            # Project C → vit_hidden
            proj = nn.Linear(C, self.vit_hidden, bias=False).to(pixel_values.device)
            frame_features = proj(frame_features).view(B, F, self.vit_hidden)

        # ---- Project to embed_dim ----------------------------------------
        frame_emb = self.visual_proj(frame_features)  # (B, F, embed_dim)

        # ---- Temporal attention pooling -----------------------------------
        attn_weights = self.temporal_attn(frame_emb)  # (B, F, 1)

        if frame_mask is not None:
            mask = frame_mask.unsqueeze(-1).float()  # (B, F, 1)
            # Zero out masked positions before softmax
            attn_weights = attn_weights * mask
            attn_sum = attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
            attn_weights = attn_weights / attn_sum

        visual_pooled = (frame_emb * attn_weights).sum(dim=1)  # (B, embed_dim)

        # ---- Behavioural features ----------------------------------------
        if behavioral_features is not None:
            beh_emb = self.behavioral_proj(behavioral_features)  # (B, embed_dim)
        else:
            beh_emb = torch.zeros_like(visual_pooled)

        # ---- Fusion: residual add ----------------------------------------
        fused = self.fusion_norm(visual_pooled + beh_emb)
        fused = self.dropout(fused)

        # ---- Output projection -------------------------------------------
        output = self.output_proj(fused)  # (B, embed_dim)
        return output
