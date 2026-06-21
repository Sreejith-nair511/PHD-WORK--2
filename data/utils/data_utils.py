"""
Data utilities for DG-HMCF.

Provides:
  - ModalityMask dataclass
  - collate_fn for variable-modality batches
  - normalize_phq8 / denormalize_phq8
  - split_dataset helper
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


# ---------------------------------------------------------------------------
# ModalityMask
# ---------------------------------------------------------------------------

@dataclass
class ModalityMask:
    """Boolean flags indicating which modalities are present in a sample."""

    has_speech: bool = False
    has_text: bool = False
    has_face: bool = False
    has_eeg: bool = False

    def to_tensor(self) -> torch.Tensor:
        """Return a float32 tensor of shape (4,): [speech, text, face, eeg]."""
        return torch.tensor(
            [
                float(self.has_speech),
                float(self.has_text),
                float(self.has_face),
                float(self.has_eeg),
            ],
            dtype=torch.float32,
        )

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "ModalityMask":
        arr = t.cpu().numpy().astype(bool)
        return cls(
            has_speech=bool(arr[0]),
            has_text=bool(arr[1]),
            has_face=bool(arr[2]),
            has_eeg=bool(arr[3]),
        )

    def __repr__(self) -> str:  # pragma: no cover
        flags = {
            "speech": self.has_speech,
            "text": self.has_text,
            "face": self.has_face,
            "eeg": self.has_eeg,
        }
        present = [k for k, v in flags.items() if v]
        return f"ModalityMask({', '.join(present)})"


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_phq8(score: float, min_val: float = 0.0, max_val: float = 24.0) -> float:
    """
    Normalise a raw PHQ-8 score to [0, 1].

    Parameters
    ----------
    score   : raw PHQ-8 score (0–24).
    min_val : minimum of the scale (default 0).
    max_val : maximum of the scale (default 24).
    """
    score = float(np.clip(score, min_val, max_val))
    return (score - min_val) / (max_val - min_val)


def denormalize_phq8(
    normalized: float, min_val: float = 0.0, max_val: float = 24.0
) -> float:
    """Convert a normalised score back to the raw PHQ-8 range."""
    return float(normalized) * (max_val - min_val) + min_val


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """
    Randomly split a Dataset into train / val / test subsets.

    Parameters
    ----------
    dataset     : PyTorch Dataset to split.
    train_ratio : fraction for training set.
    val_ratio   : fraction for validation set.
    seed        : random seed for reproducibility.

    Returns
    -------
    (train_subset, val_subset, test_subset)
    """
    n = len(dataset)  # type: ignore[arg-type]
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train: n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


# ---------------------------------------------------------------------------
# Custom collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function that handles:
      - Variable modality availability (absent modalities → None → zero tensors)
      - Mixed numpy arrays / tensors
      - ModalityMask objects

    Parameters
    ----------
    batch : list of sample dicts returned by a BaseDepressionDataset.

    Returns
    -------
    Collated dict ready for model forward pass.
    """
    collated: Dict[str, Any] = {}

    # Keys present in at least one sample
    all_keys = set()
    for sample in batch:
        all_keys.update(sample.keys())

    for key in all_keys:
        values = [s.get(key, None) for s in batch]

        # ---- modality_mask ------------------------------------------------
        if key == "modality_mask":
            # Each value is a ModalityMask or tensor (4,)
            tensors = []
            for v in values:
                if v is None:
                    tensors.append(torch.zeros(4, dtype=torch.float32))
                elif isinstance(v, ModalityMask):
                    tensors.append(v.to_tensor())
                elif isinstance(v, torch.Tensor):
                    tensors.append(v.float())
                else:
                    tensors.append(torch.tensor(v, dtype=torch.float32))
            collated[key] = torch.stack(tensors, dim=0)  # (B, 4)
            continue

        # ---- scalar targets (phq8_score, label) ---------------------------
        if key in ("phq8_score", "label"):
            tensors = []
            for v in values:
                if v is None:
                    tensors.append(torch.tensor(0.0))
                elif isinstance(v, torch.Tensor):
                    tensors.append(v.float())
                else:
                    tensors.append(torch.tensor(float(v), dtype=torch.float32))
            collated[key] = torch.stack(tensors, dim=0)  # (B,)
            continue

        # ---- string fields (subject_id) -----------------------------------
        if key == "subject_id":
            collated[key] = [str(v) if v is not None else "" for v in values]
            continue

        # ---- nested speech dict -------------------------------------------
        if key == "speech":
            collated[key] = _collate_speech(values)
            continue

        # ---- nested text dict ---------------------------------------------
        if key == "text":
            collated[key] = _collate_text(values)
            continue

        # ---- nested face dict ---------------------------------------------
        if key == "face":
            collated[key] = _collate_face(values)
            continue

        # ---- nested eeg dict ----------------------------------------------
        if key == "eeg":
            collated[key] = _collate_eeg(values)
            continue

        # ---- generic tensor/array -----------------------------------------
        collated[key] = _stack_generic(values)

    return collated


# ---------------------------------------------------------------------------
# Modality-specific collation helpers
# ---------------------------------------------------------------------------

def _to_tensor(v: Any, dtype=torch.float32) -> Optional[torch.Tensor]:
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return v.to(dtype)
    if isinstance(v, np.ndarray):
        return torch.from_numpy(v.copy()).to(dtype)
    return torch.tensor(v, dtype=dtype)


def _collate_speech(values: List[Optional[Dict]]) -> Dict[str, torch.Tensor]:
    """Collate a list of speech dicts, zero-filling absent modalities."""
    # Determine shapes from first non-None sample
    ref = next((v for v in values if v is not None), None)

    if ref is None:
        return {
            "waveform": torch.zeros(len(values), 1),
            "attention_mask": torch.zeros(len(values), 1),
            "behavioral_features": torch.zeros(len(values), 6),
        }

    max_len = ref["waveform"].shape[-1] if isinstance(ref["waveform"], (np.ndarray, torch.Tensor)) else 160000
    bf_dim = ref["behavioral_features"].shape[-1] if ref.get("behavioral_features") is not None else 6

    waveforms, masks, bfs = [], [], []
    for v in values:
        if v is None:
            waveforms.append(torch.zeros(max_len, dtype=torch.float32))
            masks.append(torch.zeros(max_len, dtype=torch.float32))
            bfs.append(torch.zeros(bf_dim, dtype=torch.float32))
        else:
            waveforms.append(_to_tensor(v["waveform"]).squeeze())
            masks.append(_to_tensor(v["attention_mask"]).squeeze())
            bfs.append(_to_tensor(v["behavioral_features"]).squeeze())

    return {
        "waveform": torch.stack(waveforms, dim=0),
        "attention_mask": torch.stack(masks, dim=0),
        "behavioral_features": torch.stack(bfs, dim=0),
    }


def _collate_text(values: List[Optional[Dict]]) -> Dict[str, torch.Tensor]:
    ref = next((v for v in values if v is not None), None)
    if ref is None:
        return {
            "input_ids": torch.zeros(len(values), 512, dtype=torch.long),
            "attention_mask": torch.zeros(len(values), 512),
            "linguistic_features": torch.zeros(len(values), 5),
        }

    seq_len = ref["input_ids"].shape[-1]
    lf_dim = ref["linguistic_features"].shape[-1] if ref.get("linguistic_features") is not None else 5

    ids, amasks, lfs = [], [], []
    for v in values:
        if v is None:
            ids.append(torch.zeros(seq_len, dtype=torch.long))
            amasks.append(torch.zeros(seq_len, dtype=torch.float32))
            lfs.append(torch.zeros(lf_dim, dtype=torch.float32))
        else:
            ids.append(_to_tensor(v["input_ids"], dtype=torch.long).squeeze())
            amasks.append(_to_tensor(v["attention_mask"]).squeeze())
            lfs.append(_to_tensor(v["linguistic_features"]).squeeze())

    return {
        "input_ids": torch.stack(ids, dim=0),
        "attention_mask": torch.stack(amasks, dim=0),
        "linguistic_features": torch.stack(lfs, dim=0),
    }


def _collate_face(values: List[Optional[Dict]]) -> Dict[str, torch.Tensor]:
    ref = next((v for v in values if v is not None), None)
    if ref is None:
        return {
            "pixel_values": torch.zeros(len(values), 1, 3, 224, 224),
            "frame_mask": torch.zeros(len(values), 1),
            "behavioral_features": torch.zeros(len(values), 7),
        }

    pv_shape = ref["pixel_values"].shape  # (max_frames, 3, H, W)
    bf_dim = ref["behavioral_features"].shape[-1] if ref.get("behavioral_features") is not None else 7

    pvs, fmasks, bfs = [], [], []
    for v in values:
        if v is None:
            pvs.append(torch.zeros(*pv_shape, dtype=torch.float32))
            fmasks.append(torch.zeros(pv_shape[0], dtype=torch.float32))
            bfs.append(torch.zeros(bf_dim, dtype=torch.float32))
        else:
            pvs.append(_to_tensor(v["pixel_values"]))
            fmasks.append(_to_tensor(v["frame_mask"]).squeeze())
            bfs.append(_to_tensor(v["behavioral_features"]).squeeze())

    return {
        "pixel_values": torch.stack(pvs, dim=0),
        "frame_mask": torch.stack(fmasks, dim=0),
        "behavioral_features": torch.stack(bfs, dim=0),
    }


def _collate_eeg(values: List[Optional[Dict]]) -> Dict[str, torch.Tensor]:
    ref = next((v for v in values if v is not None), None)
    if ref is None:
        return {
            "segments": torch.zeros(len(values), 1, 64, 256),
            "segment_mask": torch.zeros(len(values), 1),
        }

    seg_shape = ref["segments"].shape  # (max_segs, n_channels, seg_len)

    segs, smasks = [], []
    for v in values:
        if v is None:
            segs.append(torch.zeros(*seg_shape, dtype=torch.float32))
            smasks.append(torch.zeros(seg_shape[0], dtype=torch.float32))
        else:
            segs.append(_to_tensor(v["segments"]))
            smasks.append(_to_tensor(v["segment_mask"]).squeeze())

    return {
        "segments": torch.stack(segs, dim=0),
        "segment_mask": torch.stack(smasks, dim=0),
    }


def _stack_generic(values: List[Any]) -> Any:
    """Attempt to stack generic values into a tensor."""
    non_none = [v for v in values if v is not None]
    if not non_none:
        return None
    try:
        tensors = []
        for v in values:
            if v is None:
                # Use shape of first non-None
                ref_t = _to_tensor(non_none[0])
                tensors.append(torch.zeros_like(ref_t))
            else:
                tensors.append(_to_tensor(v))
        return torch.stack(tensors, dim=0)
    except Exception:
        return values
