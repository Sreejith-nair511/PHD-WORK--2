"""
Missing Modality Handler for DG-HMCF.

Generates proxy embeddings for absent modalities so the rest of the
pipeline can operate on a complete set of four modality representations.

Also provides a ModalityDropout training augmentation module.
"""

import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class MissingModalityHandler(nn.Module):
    """
    Fill in proxy embeddings for absent modalities.

    For each absent modality, the handler produces a proxy embedding via:
        proxy = learned_zero_vector + noise * is_training

    The learned_zero_vector is a trainable parameter that captures a
    "modality-absent" prior.  During training, small Gaussian noise is
    added to encourage robustness.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of each modality embedding.
    n_modalities : int
        Number of modalities (4: speech, text, face, EEG).
    noise_std : float
        Standard deviation of the training noise injected into proxy vectors.
    """

    MODALITY_ORDER = ["speech", "text", "face", "eeg"]

    def __init__(
        self,
        embed_dim: int = 256,
        n_modalities: int = 4,
        noise_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.noise_std = noise_std

        # Learnable "absent modality" priors – one per modality
        self.proxy_vectors = nn.ParameterList(
            [nn.Parameter(torch.randn(embed_dim) * 0.02) for _ in range(n_modalities)]
        )

        # Small MLP to refine proxy based on available modality context
        # Input: mean of all present embeddings; output: delta to add to proxy
        self.proxy_refiner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        embeddings: Dict[str, Optional[torch.Tensor]],
        modality_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        embeddings : dict
            Keys: 'speech', 'text', 'face', 'eeg'.
            Values: tensors of shape (B, embed_dim) or None.
        modality_mask : torch.Tensor, shape (B, 4)
            Binary availability flags.

        Returns
        -------
        complete_embeddings : dict
            All four modality embeddings present (proxies substituted for absent).
        """
        B = modality_mask.size(0)
        device = modality_mask.device

        # Compute context from present modalities
        present_embs: List[torch.Tensor] = []
        for name in self.MODALITY_ORDER:
            if embeddings.get(name) is not None:
                present_embs.append(embeddings[name])

        if present_embs:
            context = torch.stack(present_embs, dim=0).mean(dim=0)  # (B, embed_dim)
            context_delta = self.proxy_refiner(context)              # (B, embed_dim)
        else:
            context_delta = torch.zeros(B, self.embed_dim, device=device)

        complete: Dict[str, torch.Tensor] = {}

        for i, name in enumerate(self.MODALITY_ORDER):
            emb = embeddings.get(name)
            mask_i = modality_mask[:, i]  # (B,) float 0/1

            # Build proxy for this modality
            proxy_base = self.proxy_vectors[i].unsqueeze(0).expand(B, -1)  # (B, embed_dim)
            proxy = proxy_base + context_delta

            if self.training and self.noise_std > 0:
                noise = torch.randn_like(proxy) * self.noise_std
                proxy = proxy + noise

            if emb is not None:
                # Use real embedding where mask = 1, proxy where mask = 0
                mask_expand = mask_i.unsqueeze(-1)  # (B, 1)
                complete[name] = emb * mask_expand + proxy * (1.0 - mask_expand)
            else:
                # All samples for this modality are absent
                complete[name] = proxy

        return complete


class ModalityDropout(nn.Module):
    """
    Training-time data augmentation: randomly masks entire modalities.

    Simulates missing modality scenarios so the model learns to be
    robust to arbitrary subsets of available modalities.

    Parameters
    ----------
    drop_prob : float
        Probability of dropping any individual modality during training.
    min_modalities : int
        Ensure at least this many modalities remain after dropout (≥ 1).
    """

    MODALITY_ORDER = ["speech", "text", "face", "eeg"]

    def __init__(self, drop_prob: float = 0.2, min_modalities: int = 1) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.min_modalities = min_modalities

    def forward(
        self,
        embeddings: Dict[str, Optional[torch.Tensor]],
        modality_mask: torch.Tensor,
    ):
        """
        Randomly zero-out modalities during training.

        Parameters
        ----------
        embeddings    : dict of modality → tensor or None
        modality_mask : (B, 4) float tensor

        Returns
        -------
        (embeddings, modality_mask) with dropped modalities zeroed out.
        """
        if not self.training:
            return embeddings, modality_mask

        B = modality_mask.size(0)
        device = modality_mask.device

        augmented_mask = modality_mask.clone()
        augmented_embs = dict(embeddings)

        for b in range(B):
            present_ids = [
                i for i in range(4) if modality_mask[b, i].item() > 0.5
            ]
            if len(present_ids) <= self.min_modalities:
                continue  # Cannot drop further

            # Shuffle and potentially drop each modality
            random.shuffle(present_ids)
            n_can_drop = len(present_ids) - self.min_modalities
            for idx in present_ids:
                if n_can_drop <= 0:
                    break
                if random.random() < self.drop_prob:
                    augmented_mask[b, idx] = 0.0
                    n_can_drop -= 1

        # Zero out dropped modality embeddings for affected samples
        for i, name in enumerate(self.MODALITY_ORDER):
            if augmented_embs.get(name) is not None:
                mask_i = augmented_mask[:, i].unsqueeze(-1)  # (B, 1)
                augmented_embs[name] = augmented_embs[name] * mask_i

        return augmented_embs, augmented_mask
