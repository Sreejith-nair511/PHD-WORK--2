"""
MODMA Dataset implementation for DG-HMCF.

MODMA (Multi-modal Open Dataset for Mental-disorder Analysis) contains
speech and EEG recordings from patients and healthy controls.

Expected directory layout::

    <root_dir>/
        metadata.csv          # columns: subject_id, label, phq8_score (optional)
        speech/
            <subject_id>.wav
        eeg/
            <subject_id>.npy  # shape (n_channels, n_timepoints)
"""

import os
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from data.datasets.base_dataset import BaseDepressionDataset
from data.preprocessing.speech_preprocessor import SpeechPreprocessor
from data.preprocessing.eeg_preprocessor import EEGPreprocessor

logger = logging.getLogger(__name__)


class MODMADataset(BaseDepressionDataset):
    """
    PyTorch Dataset for the MODMA multi-modal mental disorder corpus.

    Supports speech and EEG modalities.

    Parameters
    ----------
    root_dir : str
        Path to the MODMA root directory.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    phq8_threshold : int
        Binary label threshold (default 10).
    modalities : list of str, optional
        Subset of ``["speech", "eeg"]`` to load.
    eeg_n_channels : int
        Number of EEG channels expected.
    eeg_sampling_rate : int
        EEG sampling rate in Hz.
    augment : bool
        Apply data augmentation during training.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        phq8_threshold: int = 10,
        modalities: Optional[List[str]] = None,
        eeg_n_channels: int = 64,
        eeg_sampling_rate: int = 256,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.speech_preprocessor = SpeechPreprocessor(
            sample_rate=16000,
            max_length=160000,
            remove_interviewer=False,  # MODMA recordings are participant-only
        )
        self.eeg_preprocessor = EEGPreprocessor(
            sampling_rate=eeg_sampling_rate,
            n_channels=eeg_n_channels,
            segment_length=256,
            max_segments=100,
            overlap=0.5,
        )

        if modalities is None:
            modalities = ["speech", "eeg"]
        modalities = [m for m in modalities if m in ("speech", "eeg")]

        super().__init__(
            root_dir=root_dir,
            split=split,
            phq8_threshold=phq8_threshold,
            modalities=modalities,
            augment=augment,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # BaseDepressionDataset interface
    # ------------------------------------------------------------------

    def load_metadata(self) -> None:
        """
        Load subject metadata.  Expects a ``metadata.csv`` in root_dir
        with at minimum columns ``subject_id`` and ``label`` (0/1).
        An optional ``phq8_score`` column is used when present.
        """
        meta_path = os.path.join(self.root_dir, "metadata.csv")
        if not os.path.exists(meta_path):
            logger.warning("MODMA metadata.csv not found at %s.", meta_path)
            self.metadata = pd.DataFrame(
                columns=["subject_id", "phq8_score", "label",
                         "speech_path", "eeg_path"]
            )
            return

        df = pd.read_csv(meta_path, dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]

        rows: List[Dict] = []
        for _, row in df.iterrows():
            sid = str(row.get("subject_id", row.get("id", ""))).strip()
            label = int(float(row.get("label", 0)))
            # PHQ-8 may not be present; simulate from label if absent
            if "phq8_score" in row and pd.notna(row["phq8_score"]):
                phq8 = float(row["phq8_score"])
            else:
                # Synthetic: depressed → 15, non-depressed → 5
                phq8 = 15.0 if label == 1 else 5.0

            speech_path = os.path.join(self.root_dir, "speech", f"{sid}.wav")
            eeg_path = os.path.join(self.root_dir, "eeg", f"{sid}.npy")

            rows.append({
                "subject_id": sid,
                "phq8_score": phq8,
                "label": label,
                "speech_path": speech_path if os.path.exists(speech_path) else None,
                "eeg_path": eeg_path if os.path.exists(eeg_path) else None,
            })

        full_df = pd.DataFrame(rows)

        # Simple deterministic train/val/test split by subject index
        full_df = full_df.reset_index(drop=True)
        n = len(full_df)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        split_map = {
            "train": idx[:n_train],
            "val": idx[n_train: n_train + n_val],
            "dev": idx[n_train: n_train + n_val],
            "test": idx[n_train + n_val:],
        }
        selected_idx = split_map.get(self.split.lower(), idx[:n_train])
        self.metadata = full_df.iloc[selected_idx].reset_index(drop=True)

        logger.info(
            "Loaded MODMA %s split: %d samples (%d depressed).",
            self.split,
            len(self.metadata),
            int(self.metadata["label"].sum()),
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[idx]
        sid = str(row["subject_id"])
        phq8_raw = float(row["phq8_score"])
        label = int(row["label"])

        # ---- Speech -------------------------------------------------------
        speech_data: Optional[Dict] = None
        if "speech" in self.modalities and row.get("speech_path"):
            try:
                speech_data = self.speech_preprocessor.preprocess(
                    audio_path=row["speech_path"],
                )
            except Exception as exc:
                logger.debug("Failed to load speech for %s: %s", sid, exc)

        # ---- EEG ----------------------------------------------------------
        eeg_data: Optional[Dict] = None
        if "eeg" in self.modalities and row.get("eeg_path"):
            try:
                raw_eeg = np.load(row["eeg_path"])
                eeg_data = self.eeg_preprocessor.preprocess(raw_eeg)
            except Exception as exc:
                logger.debug("Failed to load EEG for %s: %s", sid, exc)

        modality_mask = self._make_modality_mask(
            speech=speech_data,
            text=None,
            face=None,
            eeg=eeg_data,
        )

        return {
            "speech": speech_data,
            "text": None,
            "face": None,
            "eeg": eeg_data,
            "phq8_score": np.float32(self._normalize_phq8(phq8_raw)),
            "phq8_score_raw": np.float32(phq8_raw),
            "label": np.int64(label),
            "modality_mask": modality_mask,
            "subject_id": sid,
        }
