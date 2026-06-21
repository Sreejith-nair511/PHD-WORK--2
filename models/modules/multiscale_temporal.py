"""
Multi-Scale Temporal Fusion module for DG-HMCF.

Applies parallel dilated 1-D convolutions with different kernel sizes to
capture temporal patterns at multiple timescales simultaneously.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTemporalFusion(nn.Module):
    """
    Multi-scale temporal feature extractor.

    Applies ``len(kernel_sizes)`` parallel Conv1d branches with different
    receptive fields, then concatenates their outputs along the channel
    dimension.

    Input:  (B, seq_len, in_channels)
    Output: (B, seq_len, out_channels * len(kernel_sizes))

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output channels *per branch* (total = out_channels * len(kernel_sizes)).
    kernel_sizes : list of int
        Kernel sizes for the parallel convolution branches.
    dropout : float
        Dropout probability applied after each branch.
    use_residual : bool
        Add a residual projection from input to output when dimensions match.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 64,
        kernel_sizes: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        self.total_out = out_channels * len(kernel_sizes)

        # Parallel convolution branches
        self.conv_branches = nn.ModuleList()
        for k in kernel_sizes:
            branch = nn.Sequential(
                # Same-padding so output length == input length
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.conv_branches.append(branch)

        # Channel-wise mixing across branches
        self.channel_mix = nn.Sequential(
            nn.Conv1d(
                in_channels=self.total_out,
                out_channels=self.total_out,
                kernel_size=1,
                bias=True,
            ),
            nn.BatchNorm1d(self.total_out),
            nn.GELU(),
        )

        # Residual projection (if enabled and dimensions match)
        self.use_residual = use_residual and (in_channels == self.total_out)
        if use_residual and in_channels != self.total_out:
            self.residual_proj = nn.Conv1d(in_channels, self.total_out, kernel_size=1, bias=False)
        else:
            self.residual_proj = None

        self.layer_norm = nn.LayerNorm(self.total_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, seq_len, in_channels)

        Returns
        -------
        torch.Tensor, shape (B, seq_len, out_channels * len(kernel_sizes))
        """
        # Conv1d expects (B, C, L)
        x_t = x.transpose(1, 2)  # (B, in_channels, seq_len)

        # Apply each branch
        branch_outputs = [branch(x_t) for branch in self.conv_branches]

        # Concatenate along channel dim
        multi_scale = torch.cat(branch_outputs, dim=1)  # (B, total_out, seq_len)

        # Channel mixing
        out = self.channel_mix(multi_scale)  # (B, total_out, seq_len)

        # Residual connection
        if self.use_residual:
            out = out + x_t
        elif self.residual_proj is not None:
            out = out + self.residual_proj(x_t)

        # Back to (B, seq_len, total_out)
        out = out.transpose(1, 2)  # (B, seq_len, total_out)
        out = self.layer_norm(out)
        return out


# Make Optional import visible at module level for the type annotation above
from typing import Optional  # noqa: E402 (placed after to avoid circular issues)
