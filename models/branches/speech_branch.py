"""
Speech Branch for DG-HMCF.

Combines a Wav2Vec2 acoustic backbone with a behavioural feature MLP
and a BiLSTM temporal encoder to produce a fixed-size speech embedding.
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from transformers import Wav2Vec2Model
except ImportError:
    Wav2Vec2Model = None  # type: ignore


class SpeechBranch(nn.Module):
    """
    Speech encoder branch.

    Architecture:
      1. Wav2Vec2 encoder  → (B, T, 768)
      2. Linear projection → (B, T, embed_dim)
      3. Concatenate per-token behavioural features (broadcast)
      4. BiLSTM                → (B, T, 2*bilstm_hidden)
      5. Mean-pool over T      → (B, embed_dim)

    Parameters
    ----------
    wav2vec2_model : str
        HuggingFace model identifier (e.g., ``"facebook/wav2vec2-base-960h"``).
    behavioral_feat_dim : int
        Dimensionality of the behavioural feature vector.
    embed_dim : int
        Output embedding dimension.
    bilstm_hidden : int
        BiLSTM hidden units per direction.
    bilstm_layers : int
        Number of BiLSTM layers.
    dropout : float
        Dropout probability.
    freeze_wav2vec2 : bool
        If True, freeze Wav2Vec2 weights and only train downstream modules.
    """

    def __init__(
        self,
        wav2vec2_model: str = "facebook/wav2vec2-base-960h",
        behavioral_feat_dim: int = 6,
        embed_dim: int = 256,
        bilstm_hidden: int = 128,
        bilstm_layers: int = 2,
        dropout: float = 0.1,
        freeze_wav2vec2: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.wav2vec2_hidden = 768  # Wav2Vec2-base hidden size

        # ---- Wav2Vec2 backbone --------------------------------------------
        if Wav2Vec2Model is not None:
            try:
                self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec2_model)
                if freeze_wav2vec2:
                    for param in self.wav2vec2.parameters():
                        param.requires_grad = False
            except Exception:
                self.wav2vec2 = None
        else:
            self.wav2vec2 = None

        # ---- Behavioural feature MLP --------------------------------------
        # Projects scalar features to embed_dim for concatenation
        self.behavioral_proj = nn.Sequential(
            nn.Linear(behavioral_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- Acoustic projection ------------------------------------------
        self.acoustic_proj = nn.Sequential(
            nn.Linear(self.wav2vec2_hidden, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # ---- BiLSTM temporal encoder -------------------------------------
        # Input dimension = acoustic embed + behavioural embed
        bilstm_input_dim = embed_dim + embed_dim
        self.bilstm = nn.LSTM(
            input_size=bilstm_input_dim,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if bilstm_layers > 1 else 0.0,
        )

        # ---- Output projection -------------------------------------------
        self.output_proj = nn.Sequential(
            nn.Linear(bilstm_hidden * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        behavioral_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_values : torch.Tensor, shape (B, max_length)
            Raw waveform samples.
        attention_mask : torch.Tensor, shape (B, max_length), optional
            1 for real samples, 0 for padding.
        behavioral_features : torch.Tensor, shape (B, behavioral_feat_dim), optional
            Pre-computed behavioural feature vector.

        Returns
        -------
        torch.Tensor, shape (B, embed_dim)
            Speech embedding.
        """
        batch_size = input_values.size(0)

        # ---- Acoustic encoding -------------------------------------------
        if self.wav2vec2 is not None:
            with torch.set_grad_enabled(True):
                outputs = self.wav2vec2(
                    input_values=input_values,
                    attention_mask=attention_mask,
                )
            hidden_states = outputs.last_hidden_state  # (B, T, 768)
        else:
            # Fallback: simple random projection of waveform segments
            # Reshape waveform into 20-ms frames (320 samples at 16 kHz)
            T = input_values.size(1) // 320
            if T == 0:
                T = 1
            hidden_states = input_values[:, : T * 320].view(batch_size, T, 320)
            # Project 320 → 768
            proj = nn.Linear(320, 768, bias=False).to(input_values.device)
            hidden_states = proj(hidden_states)

        # Project to embed_dim
        acoustic_emb = self.acoustic_proj(hidden_states)  # (B, T, embed_dim)

        # ---- Behavioural features -----------------------------------------
        if behavioral_features is not None:
            beh_emb = self.behavioral_proj(behavioral_features)  # (B, embed_dim)
            # Broadcast to sequence dimension
            beh_emb = beh_emb.unsqueeze(1).expand(-1, acoustic_emb.size(1), -1)
        else:
            beh_emb = torch.zeros_like(acoustic_emb)

        # ---- Concatenate and BiLSTM --------------------------------------
        combined = torch.cat([acoustic_emb, beh_emb], dim=-1)  # (B, T, 2*embed_dim)
        combined = self.dropout(combined)

        lstm_out, _ = self.bilstm(combined)  # (B, T, 2*bilstm_hidden)

        # ---- Masked mean pooling -----------------------------------------
        if attention_mask is not None:
            # Downsample attention mask to sequence length T
            T = lstm_out.size(1)
            orig_len = attention_mask.size(1)
            if orig_len != T:
                # Average-pool mask to frame level
                mask_float = attention_mask.float()
                mask_float = mask_float.unsqueeze(1)  # (B, 1, orig_len)
                mask_downsampled = torch.nn.functional.adaptive_avg_pool1d(
                    mask_float, T
                ).squeeze(1)  # (B, T)
                frame_mask = (mask_downsampled > 0.5).float()
            else:
                frame_mask = attention_mask.float()

            frame_mask = frame_mask.unsqueeze(-1)  # (B, T, 1)
            pooled = (lstm_out * frame_mask).sum(dim=1) / (
                frame_mask.sum(dim=1).clamp(min=1.0)
            )
        else:
            pooled = lstm_out.mean(dim=1)  # (B, 2*bilstm_hidden)

        # ---- Output projection -------------------------------------------
        output = self.output_proj(pooled)  # (B, embed_dim)
        return output
