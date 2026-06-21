"""
DAIC-WOZ Dataset implementation for DG-HMCF.

The DAIC-WOZ (Distress Analysis Interview Corpus – Wizard of Oz) dataset
contains clinical interviews conducted by a virtual agent ("Ellie") and
is labelled with PHQ-8 scores.

Expected directory layout::

    <root_dir>/
        train_split_Depression_AVEC2017.csv
        dev_split_Depression_AVEC2017.csv
        test_split_Depression_AVEC2017.csv
        <participant_id>_P/
            <participant_id>_AUDIO.wav
            <participant_id>_TRANSCRIPT.csv
            <participant_id>_CLNF_features3D.txt   (optional)
            <participant_id>_FORMANT.csv            (optional)
"""

import os
import csv
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from data.datasets.base_dataset import BaseDepressionDataset
from data.preprocessing.speech_preprocessor import SpeechPreprocessor
from data.preprocessing.text_preprocessor import TextPreprocessor
from data.preprocessing.face_preprocessor import FacePreprocessor

logger = logging.getLogger(__name__)

# Mapping from AVEC split name to CSV filename
_SPLIT_CSV = {
    "train": "train_split_Depression_AVEC2017.csv",
    "dev": "dev_split_Depression_AVEC2017.csv",
    "val": "dev_split_Depression_AVEC2017.csv",  # alias
    "test": "test_split_Depression_AVEC2017.csv",
}


class DAICWOZDataset(BaseDepressionDataset):
    """
    PyTorch Dataset for the DAIC-WOZ depression corpus.

    Supports speech, text, and face modalities.  EEG is not available
    in DAIC-WOZ, so ``has_eeg`` will always be False.

    Parameters
    ----------
    root_dir : str
        Path to the DAIC-WOZ root directory.
    split : str
        One of ``"train"``, ``"dev"``/``"val"``, ``"test"``.
    phq8_threshold : int
        Binary label threshold (default 10).
    modalities : list of str, optional
        Subset of ``["speech", "text", "face"]`` to load.
    augment : bool
        Apply data augmentation during training.
    speech_max_length : int
        Max waveform length in samples (default 160 000 = 10 s).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        phq8_threshold: int = 10,
        modalities: Optional[List[str]] = None,
        augment: bool = False,
        speech_max_length: int = 160000,
        seed: int = 42,
    ) -> None:
        # Preprocessors
        self.speech_preprocessor = SpeechPreprocessor(
            sample_rate=16000,
            max_length=speech_max_length,
            remove_interviewer=True,
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

        if modalities is None:
            modalities = ["speech", "text", "face"]
        # Remove EEG – not available in DAIC-WOZ
        modalities = [m for m in modalities if m != "eeg"]

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
        Load participant metadata from the AVEC 2017 CSV files.

        Creates ``self.metadata`` with columns:
        participant_id, phq8_score, label, audio_path, transcript_path,
        video_path (if available).
        """
        csv_name = _SPLIT_CSV.get(self.split.lower())
        if csv_name is None:
            logger.warning("Unknown split %r – defaulting to train.", self.split)
            csv_name = _SPLIT_CSV["train"]

        csv_path = os.path.join(self.root_dir, csv_name)
        if not os.path.exists(csv_path):
            logger.warning("Metadata CSV not found: %s. Using empty dataset.", csv_path)
            self.metadata = pd.DataFrame(
                columns=["participant_id", "phq8_score", "label",
                         "audio_path", "transcript_path", "video_path"]
            )
            return

        rows: List[Dict] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = str(row.get("Participant_ID", row.get("participant_id", ""))).strip()
                try:
                    phq8 = float(row.get("PHQ8_Score", row.get("phq8_score", 0)))
                except ValueError:
                    phq8 = 0.0

                participant_dir = os.path.join(self.root_dir, f"{pid}_P")
                audio_path = os.path.join(participant_dir, f"{pid}_AUDIO.wav")
                transcript_path = os.path.join(participant_dir, f"{pid}_TRANSCRIPT.csv")
                video_path = os.path.join(participant_dir, f"{pid}_VIDEO.mp4")

                rows.append({
                    "participant_id": pid,
                    "phq8_score": phq8,
                    "label": self._make_label(phq8),
                    "audio_path": audio_path if os.path.exists(audio_path) else None,
                    "transcript_path": transcript_path if os.path.exists(transcript_path) else None,
                    "video_path": video_path if os.path.exists(video_path) else None,
                })

        self.metadata = pd.DataFrame(rows)
        logger.info(
            "Loaded DAIC-WOZ %s split: %d samples (%d depressed).",
            self.split,
            len(self.metadata),
            int(self.metadata["label"].sum()),
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[idx]
        pid = str(row["participant_id"])
        phq8_raw = float(row["phq8_score"])
        label = int(row["label"])

        # ---- Speech -------------------------------------------------------
        speech_data: Optional[Dict] = None
        if "speech" in self.modalities and row.get("audio_path"):
            try:
                speech_data = self.speech_preprocessor.preprocess(
                    audio_path=row["audio_path"],
                    transcript_path=row.get("transcript_path"),
                )
            except Exception as exc:
                logger.debug("Failed to load speech for %s: %s", pid, exc)

        # ---- Text ---------------------------------------------------------
        text_data: Optional[Dict] = None
        if "text" in self.modalities and row.get("transcript_path"):
            try:
                transcript = self._load_daic_transcript(row["transcript_path"])
                text_data = self.text_preprocessor.preprocess(
                    transcript=transcript,
                    remove_interviewer=True,
                )
            except Exception as exc:
                logger.debug("Failed to load text for %s: %s", pid, exc)

        # ---- Face ---------------------------------------------------------
        face_data: Optional[Dict] = None
        if "face" in self.modalities and row.get("video_path"):
            try:
                face_data = self.face_preprocessor.preprocess(row["video_path"])
            except Exception as exc:
                logger.debug("Failed to load face for %s: %s", pid, exc)

        modality_mask = self._make_modality_mask(
            speech=speech_data,
            text=text_data,
            face=face_data,
            eeg=None,
        )

        return {
            "speech": speech_data,
            "text": text_data,
            "face": face_data,
            "eeg": None,
            "phq8_score": np.float32(self._normalize_phq8(phq8_raw)),
            "phq8_score_raw": np.float32(phq8_raw),
            "label": np.int64(label),
            "modality_mask": modality_mask,
            "subject_id": pid,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_daic_transcript(transcript_path: str) -> str:
        """
        Load and concatenate participant turns from a DAIC-WOZ transcript CSV.

        The DAIC-WOZ transcript CSV has columns:
        speaker_role, start_time, stop_time, value
        """
        lines: List[str] = []
        try:
            df = pd.read_csv(transcript_path, sep="\t", header=None,
                             names=["start", "stop", "speaker", "value"],
                             dtype=str)
        except Exception:
            try:
                df = pd.read_csv(transcript_path, dtype=str)
            except Exception:
                return ""

        # Try to identify speaker and text columns
        if "speaker" in df.columns and "value" in df.columns:
            for _, row in df.iterrows():
                speaker = str(row.get("speaker", "")).strip().lower()
                text = str(row.get("value", "")).strip()
                if speaker and text and speaker != "ellie":
                    lines.append(text)
        elif len(df.columns) >= 2:
            # Assume last column is text
            for _, row in df.iterrows():
                text = str(row.iloc[-1]).strip()
                if text and text.lower() not in ("nan", ""):
                    lines.append(text)

        return " ".join(lines)
