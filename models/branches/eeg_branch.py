"""
EEG Branch for DG-HMCF.

1-D convolutional feature extractor followed by a BiLSTM temporal encoder
for EEG segment classification.
"""

from typing import List, Optional

import torch
import torch.nn as nn


class EEGBranch(nn.Module):
    """
    EEG encoder branch.

    Architecture:
      1. Stack of [Conv1d → BatchNorm → ReLU → MaxPool] blocks
      2. Flatten channel × time into a feature vector per segment
      3. BiLSTM over segments
      4. Masked mean-pool over segments
      5. Linear projection to embed_dim

    Parameters
    ----------
    n_channels : int
        Number of EEG electrode channels.
    segment_length : int
        Number of time-points per segment.
    cnn_channels : list of int
        Number of output channels for each CNN block.
    embed_dim : int
        Output embedding dimension.
    bilstm_hidden : int
        BiLSTM hidden units per direction.
    bilstm_layers : int
        Number of stacked BiLSTM layers.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_channels: int = 64,
        segment_length: int = 256,
        cnn_channels: Optional[List[int]] = None,
        embed_dim: int = 256,
        bilstm_hidden: int = 128,
        bilstm_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        if cnn_channels is None:
            cnn_channels = [64, 128, 256]

        # ---- 1-D CNN temporal feature extractor --------------------------
        cnn_blocks: List[nn.Module] = []
        in_ch = n_channels
        for out_ch in cnn_channels:
            block = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            )
            cnn_blocks.append(block)
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_blocks)

        # Compute CNN output length (segment_length after pooling)
        cnn_out_len = segment_length
        for _ in cnn_channels:
            cnn_out_len = cnn_out_len // 2

        cnn_out_features = cnn_channels[-1] * max(cnn_out_len, 1)

        # ---- CNN output projection to BiLSTM input -----------------------
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- BiLSTM over segments ----------------------------------------
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
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
        eeg_segments: torch.Tensor,
        segment_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        eeg_segments : torch.Tensor, shape (B, S, n_channels, segment_length)
            Pre-processed and segmented EEG data.
        segment_mask : torch.Tensor, shape (B, S), optional
            1 for real segments, 0 for padding.

        Returns
        -------
        torch.Tensor, shape (B, embed_dim)
        """
        B, S, C, L = eeg_segments.shape

        # ---- CNN over each segment ---------------------------------------
        # Flatten batch and segment dimensions
        segs_flat = eeg_segments.view(B * S, C, L)       # (B*S, C, L)
        cnn_out = self.cnn(segs_flat)                     # (B*S, cnn_ch[-1], L')

        # Flatten spatial → feature vector
        cnn_flat = cnn_out.view(B * S, -1)               # (B*S, cnn_ch[-1] * L')

        # Project to embed_dim
        seg_emb = self.cnn_proj(cnn_flat)                # (B*S, embed_dim)
        seg_emb = seg_emb.view(B, S, self.embed_dim)     # (B, S, embed_dim)
        seg_emb = self.dropout(seg_emb)

        # ---- BiLSTM over segment sequence --------------------------------
        lstm_out, _ = self.bilstm(seg_emb)               # (B, S, 2*bilstm_hidden)

        # ---- Masked mean pooling -----------------------------------------
        if segment_mask is not None:
            mask = segment_mask.float().unsqueeze(-1)    # (B, S, 1)
            pooled = (lstm_out * mask).sum(dim=1) / (
                mask.sum(dim=1).clamp(min=1.0)
            )
        else:
            pooled = lstm_out.mean(dim=1)                # (B, 2*bilstm_hidden)

        # ---- Output projection -------------------------------------------
        output = self.output_proj(pooled)                # (B, embed_dim)
        return output
