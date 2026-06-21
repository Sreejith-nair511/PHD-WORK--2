"""
Text Branch for DG-HMCF.

Combines a RoBERTa language model backbone with psycholinguistic feature
injection and a BiLSTM temporal encoder.
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from transformers import RobertaModel
except ImportError:
    RobertaModel = None  # type: ignore


class TextBranch(nn.Module):
    """
    Text encoder branch.

    Architecture:
      1. RoBERTa encoder          → (B, T, 768)
      2. Linear projection        → (B, T, embed_dim)
      3. Linguistic feature MLP   → (B, embed_dim), broadcast to (B, T, embed_dim)
      4. Element-wise add
      5. BiLSTM                   → (B, T, 2*bilstm_hidden)
      6. Masked mean-pool + [CLS] → (B, embed_dim)

    Parameters
    ----------
    roberta_model : str
        HuggingFace model identifier (e.g., ``"roberta-base"``).
    linguistic_feat_dim : int
        Dimensionality of the linguistic feature vector.
    embed_dim : int
        Output embedding dimension.
    bilstm_hidden : int
        BiLSTM hidden units per direction.
    bilstm_layers : int
        Number of BiLSTM layers.
    dropout : float
        Dropout probability.
    freeze_roberta : bool
        If True, freeze RoBERTa weights.
    """

    def __init__(
        self,
        roberta_model: str = "roberta-base",
        linguistic_feat_dim: int = 5,
        embed_dim: int = 256,
        bilstm_hidden: int = 128,
        bilstm_layers: int = 2,
        dropout: float = 0.1,
        freeze_roberta: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.roberta_hidden = 768  # RoBERTa-base hidden size

        # ---- RoBERTa backbone --------------------------------------------
        if RobertaModel is not None:
            try:
                self.roberta = RobertaModel.from_pretrained(roberta_model)
                if freeze_roberta:
                    for param in self.roberta.parameters():
                        param.requires_grad = False
            except Exception:
                self.roberta = None
        else:
            self.roberta = None

        # ---- Token-level projection ---------------------------------------
        self.token_proj = nn.Sequential(
            nn.Linear(self.roberta_hidden, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # ---- Linguistic feature MLP ---------------------------------------
        self.linguistic_proj = nn.Sequential(
            nn.Linear(linguistic_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- BiLSTM -------------------------------------------------------
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if bilstm_layers > 1 else 0.0,
        )

        # ---- Output projection -------------------------------------------
        # Combine BiLSTM mean-pool + CLS token projection
        self.output_proj = nn.Sequential(
            nn.Linear(bilstm_hidden * 2 + embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        linguistic_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : torch.Tensor, shape (B, seq_len)
        attention_mask : torch.Tensor, shape (B, seq_len), optional
        linguistic_features : torch.Tensor, shape (B, linguistic_feat_dim), optional

        Returns
        -------
        torch.Tensor, shape (B, embed_dim)
        """
        # ---- RoBERTa forward pass ----------------------------------------
        if self.roberta is not None:
            roberta_out = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            sequence_output = roberta_out.last_hidden_state  # (B, T, 768)
            cls_output = sequence_output[:, 0, :]            # (B, 768)
        else:
            # Fallback: random embedding
            B, T = input_ids.shape
            sequence_output = torch.randn(B, T, self.roberta_hidden, device=input_ids.device)
            cls_output = sequence_output[:, 0, :]

        # ---- Token projection --------------------------------------------
        token_emb = self.token_proj(sequence_output)   # (B, T, embed_dim)
        cls_emb = self.token_proj(cls_output)           # (B, embed_dim)

        # ---- Linguistic features injection --------------------------------
        if linguistic_features is not None:
            ling_emb = self.linguistic_proj(linguistic_features)  # (B, embed_dim)
            # Add to each token position (broadcast)
            ling_emb_expanded = ling_emb.unsqueeze(1).expand_as(token_emb)
            token_emb = token_emb + ling_emb_expanded
        
        token_emb = self.dropout(token_emb)

        # ---- BiLSTM over token sequence ----------------------------------
        lstm_out, _ = self.bilstm(token_emb)  # (B, T, 2*bilstm_hidden)

        # ---- Masked mean pooling -----------------------------------------
        if attention_mask is not None:
            mask = attention_mask.float().unsqueeze(-1)  # (B, T, 1)
            pooled = (lstm_out * mask).sum(dim=1) / (
                mask.sum(dim=1).clamp(min=1.0)
            )
        else:
            pooled = lstm_out.mean(dim=1)  # (B, 2*bilstm_hidden)

        # ---- Combine with CLS embedding ----------------------------------
        combined = torch.cat([pooled, cls_emb], dim=-1)  # (B, 2*bilstm_hidden + embed_dim)

        # ---- Output projection -------------------------------------------
        output = self.output_proj(combined)  # (B, embed_dim)
        return output
