"""
PDCH Dataset implementation for DG-HMCF.

PDCH (Psychiatric Depression Corpus – Hypothetical) is a full 4-modality
dataset containing speech, text, face, and EEG for each participant.

Expected directory layout::

    <root_dir>/
        metadata.csv          # subject_id, label, phq8_score, split
        speech/  <subject_id>.wav
        text/    <subject_id>.txt
        video/   <subject_id>.mp4
        eeg/     <subject_id>.npy
"""

import os
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from data.datasets.base_dataset import BaseDepressionDataset
from data.preprocessing.speech_preprocessor import SpeechPreprocessor
from data.preprocessing.text_preprocessor import TextPreprocessor
from data.preprocessing.face_preprocessor import FacePreprocessor
from data.preprocessing.eeg_preprocessor import EEGPreprocessor

logger = logging.getLogger(__name__)


class PDCHDataset(BaseDepressionDataset):
    """
    PyTorch Dataset for the PDCH 4-modality depression corpus.

    Parameters
    ----------
    root_dir : str
        Path to the PDCH root directory.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    phq8_threshold : int
        Binary label threshold.
    modalities : list of str, optional
        Which modalities to load; defaults to all four.
    eeg_n_channels : int
        Number of EEG channels in the recording.
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
            remove_interviewer=False,
        )
        self.text_preprocessor = TextPreprocessor(
            model_name="roberta-base",
            max_length=512,
        )
        self.face_preprocessor = FacePreprocessor(
            image_size=224,
            fps=30,
            max_frames=300,
        )
        self.eeg_preprocessor = EEGPreprocessor(
            sampling_rate=eeg_sampling_rate,
            n_channels=eeg_n_channels,
            segment_length=256,
            max_segments=100,
            overlap=0.5,
        )

        if modalities is None:
            modalities = ["speech", "text", "face", "eeg"]

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
        """Load metadata from ``<root_dir>/metadata.csv``."""
        meta_path = os.path.join(self.root_dir, "metadata.csv")
        if not os.path.exists(meta_path):
            logger.warning("PDCH metadata.csv not found at %s.", meta_path)
            self.metadata = pd.DataFrame(
                columns=["subject_id", "phq8_score", "label", "split",
                         "speech_path", "text_path", "video_path", "eeg_path"]
            )
            return

        df = pd.read_csv(meta_path, dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]

        rows: List[Dict] = []
        for _, row in df.iterrows():
            sid = str(row.get("subject_id", row.get("id", ""))).strip()
            label = int(float(row.get("label", 0)))
            phq8 = float(row.get("phq8_score", 15.0 if label == 1 else 5.0))
            row_split = str(row.get("split", "train")).strip().lower()

            speech_path = os.path.join(self.root_dir, "speech", f"{sid}.wav")
            text_path = os.path.join(self.root_dir, "text", f"{sid}.txt")
            video_path = os.path.join(self.root_dir, "video", f"{sid}.mp4")
            eeg_path = os.path.join(self.root_dir, "eeg", f"{sid}.npy")

            rows.append({
                "subject_id": sid,
                "phq8_score": phq8,
                "label": label,
                "split": row_split,
                "speech_path": speech_path if os.path.exists(speech_path) else None,
                "text_path": text_path if os.path.exists(text_path) else None,
                "video_path": video_path if os.path.exists(video_path) else None,
                "eeg_path": eeg_path if os.path.exists(eeg_path) else None,
            })

        full_df = pd.DataFrame(rows)

        # Filter by split column if present, else do random split
        if "split" in full_df.columns and full_df["split"].nunique() > 1:
            target = self.split.lower()
            if target == "val":
                target = "dev"
            mask = full_df["split"] == target
            if mask.any():
                self.metadata = full_df[mask].reset_index(drop=True)
            else:
                self.metadata = full_df.reset_index(drop=True)
        else:
            # Deterministic split by index
            rng = np.random.default_rng(self.seed)
            n = len(full_df)
            idx = rng.permutation(n)
            n_train, n_val = int(n * 0.70), int(n * 0.15)
            split_map = {
                "train": idx[:n_train],
                "val": idx[n_train: n_train + n_val],
                "dev": idx[n_train: n_train + n_val],
                "test": idx[n_train + n_val:],
            }
            sel = split_map.get(self.split.lower(), idx[:n_train])
            self.metadata = full_df.iloc[sel].reset_index(drop=True)

        logger.info(
            "Loaded PDCH %s split: %d samples (%d depressed).",
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
                speech_data = self.speech_preprocessor.preprocess(row["speech_path"])
            except Exception as exc:
                logger.debug("Speech load failed for %s: %s", sid, exc)

        # ---- Text ---------------------------------------------------------
        text_data: Optional[Dict] = None
        if "text" in self.modalities and row.get("text_path"):
            try:
                with open(row["text_path"], "r", encoding="utf-8") as f:
                    transcript = f.read()
                text_data = self.text_preprocessor.preprocess(
                    transcript=transcript,
                    remove_interviewer=False,
                )
            except Exception as exc:
                logger.debug("Text load failed for %s: %s", sid, exc)

        # ---- Face ---------------------------------------------------------
        face_data: Optional[Dict] = None
        if "face" in self.modalities and row.get("video_path"):
            try:
                face_data = self.face_preprocessor.preprocess(row["video_path"])
            except Exception as exc:
                logger.debug("Face load failed for %s: %s", sid, exc)

        # ---- EEG ----------------------------------------------------------
        eeg_data: Optional[Dict] = None
        if "eeg" in self.modalities and row.get("eeg_path"):
            try:
                raw_eeg = np.load(row["eeg_path"])
                eeg_data = self.eeg_preprocessor.preprocess(raw_eeg)
            except Exception as exc:
                logger.debug("EEG load failed for %s: %s", sid, exc)

        modality_mask = self._make_modality_mask(
            speech=speech_data,
            text=text_data,
            face=face_data,
            eeg=eeg_data,
        )

        return {
            "speech": speech_data,
            "text": text_data,
            "face": face_data,
            "eeg": eeg_data,
            "phq8_score": np.float32(self._normalize_phq8(phq8_raw)),
            "phq8_score_raw": np.float32(phq8_raw),
            "label": np.int64(label),
            "modality_mask": modality_mask,
            "subject_id": sid,
        }
