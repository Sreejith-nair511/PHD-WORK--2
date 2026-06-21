"""
Abstract base class for depression detection datasets.

All concrete dataset implementations must subclass BaseDepressionDataset
and implement ``__len__``, ``__getitem__``, and ``load_metadata``.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data.utils.data_utils import ModalityMask, normalize_phq8


class BaseDepressionDataset(Dataset, ABC):
    """
    Abstract base class for multi-modal depression detection datasets.

    Each sample returned by ``__getitem__`` is a dict with the following keys:

    Keys
    ----
    speech : dict or None
        Sub-dict with ``waveform``, ``attention_mask``, ``behavioral_features``.
    text : dict or None
        Sub-dict with ``input_ids``, ``attention_mask``, ``linguistic_features``.
    face : dict or None
        Sub-dict with ``pixel_values``, ``frame_mask``, ``behavioral_features``.
    eeg : dict or None
        Sub-dict with ``segments``, ``segment_mask``.
    phq8_score : float
        Normalised PHQ-8 score in [0, 1].
    phq8_score_raw : float
        Raw PHQ-8 score (0–24).
    label : int
        Binary depression label (0 = non-depressed, 1 = depressed).
    modality_mask : ModalityMask
        Flags indicating which modalities are present for this sample.
    subject_id : str
        Unique participant identifier.
    """

    PHQ8_MIN = 0
    PHQ8_MAX = 24

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        phq8_threshold: int = 10,
        modalities: Optional[List[str]] = None,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        root_dir       : path to dataset root directory.
        split          : one of ``"train"``, ``"val"``, ``"test"``.
        phq8_threshold : PHQ-8 cut-off for binary label (>= threshold → depressed).
        modalities     : list of modality names to load; None means all available.
        augment        : whether to apply data augmentation during training.
        seed           : random seed for reproducibility.
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.phq8_threshold = phq8_threshold
        self.modalities = modalities or ["speech", "text", "face", "eeg"]
        self.augment = augment and (split == "train")
        self.seed = seed

        # Metadata loaded by the subclass
        self.metadata: pd.DataFrame = pd.DataFrame()
        self.load_metadata()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples in the split."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and return a single sample.

        Must return a dict matching the schema described in the class docstring.
        """
        ...

    @abstractmethod
    def load_metadata(self) -> None:
        """
        Load metadata (labels, file paths, splits) from disk into
        ``self.metadata`` (a pandas DataFrame).
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _make_label(self, phq8_score: float) -> int:
        """Convert a raw PHQ-8 score to a binary label."""
        return int(phq8_score >= self.phq8_threshold)

    def _make_modality_mask(
        self,
        speech: Optional[Any] = None,
        text: Optional[Any] = None,
        face: Optional[Any] = None,
        eeg: Optional[Any] = None,
    ) -> ModalityMask:
        """Build a ModalityMask from optional sample dictionaries."""
        return ModalityMask(
            has_speech=(speech is not None) and ("speech" in self.modalities),
            has_text=(text is not None) and ("text" in self.modalities),
            has_face=(face is not None) and ("face" in self.modalities),
            has_eeg=(eeg is not None) and ("eeg" in self.modalities),
        )

    def _normalize_phq8(self, score: float) -> float:
        return normalize_phq8(score, self.PHQ8_MIN, self.PHQ8_MAX)

    def get_class_weights(self) -> np.ndarray:
        """
        Compute inverse-frequency class weights for imbalanced datasets.

        Returns
        -------
        np.ndarray of shape (2,): [weight_non_depressed, weight_depressed]
        """
        if self.metadata.empty or "label" not in self.metadata.columns:
            return np.array([1.0, 1.0])
        labels = self.metadata["label"].values
        classes, counts = np.unique(labels, return_counts=True)
        weights = len(labels) / (len(classes) * counts)
        weight_array = np.ones(2, dtype=np.float32)
        for cls, w in zip(classes, weights):
            weight_array[int(cls)] = float(w)
        return weight_array

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"split={self.split!r}, "
            f"n_samples={len(self)}, "
            f"modalities={self.modalities})"
        )
