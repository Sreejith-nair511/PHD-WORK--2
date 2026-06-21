#!/usr/bin/env python
"""
Data preprocessing pipeline runner for DG-HMCF.

Preprocesses raw data files and saves processed features to disk,
enabling faster DataLoader iterations during training.

Usage:
    python scripts/preprocess_data.py \
        --dataset daic_woz \
        --data_root data/raw/daic_woz \
        --output_dir data/processed/daic_woz \
        --modalities speech text face \
        --n_jobs 4

For each subject, the script creates:
    <output_dir>/<subject_id>/
        speech.npz    (waveform, attention_mask, behavioral_features)
        text.npz      (input_ids, attention_mask, linguistic_features)
        face.npz      (pixel_values, frame_mask, behavioral_features)
        eeg.npz       (segments, segment_mask)
"""

import argparse
import os
import sys
import glob
import json
import logging
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("preprocess")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess dataset for DG-HMCF")
    parser.add_argument("--dataset", type=str, default="daic_woz",
                        choices=["daic_woz", "modma", "pdch"])
    parser.add_argument("--data_root", type=str, required=True,
                        help="Raw data root directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save processed features")
    parser.add_argument("--modalities", nargs="+",
                        default=["speech", "text", "face", "eeg"],
                        choices=["speech", "text", "face", "eeg"])
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_audio_length", type=int, default=160000,
                        help="Max waveform length (samples)")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--eeg_sampling_rate", type=int, default=256)
    parser.add_argument("--eeg_n_channels", type=int, default=64)
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing processed files")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-modality preprocessing functions
# ---------------------------------------------------------------------------

def preprocess_speech(
    audio_path: str,
    transcript_path: Optional[str],
    sample_rate: int,
    max_length: int,
) -> Optional[Dict]:
    from data.preprocessing.speech_preprocessor import SpeechPreprocessor
    preprocessor = SpeechPreprocessor(
        sample_rate=sample_rate,
        max_length=max_length,
        remove_interviewer=transcript_path is not None,
    )
    try:
        result = preprocessor.preprocess(audio_path, transcript_path)
        return result
    except Exception as exc:
        logger.debug("Speech preprocessing failed for %s: %s", audio_path, exc)
        return None


def preprocess_text(
    transcript_path: Optional[str],
    text_content: Optional[str] = None,
    max_length: int = 512,
) -> Optional[Dict]:
    from data.preprocessing.text_preprocessor import TextPreprocessor
    preprocessor = TextPreprocessor(max_length=max_length)

    text = text_content
    if text is None and transcript_path and os.path.exists(transcript_path):
        # Try to load plain text
        if transcript_path.endswith(".txt"):
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                return None
        elif transcript_path.endswith(".csv"):
            # DAIC-WOZ style
            try:
                import pandas as pd
                df = pd.read_csv(transcript_path, dtype=str, sep="\t",
                                 header=None, names=["start", "stop", "speaker", "value"])
                participant_lines = df[df["speaker"].str.lower() != "ellie"]["value"].dropna()
                text = " ".join(participant_lines.tolist())
            except Exception:
                return None

    if not text or not text.strip():
        return None

    try:
        return preprocessor.preprocess(text, remove_interviewer=True)
    except Exception as exc:
        logger.debug("Text preprocessing failed: %s", exc)
        return None


def preprocess_face(
    video_path: str,
    image_size: int,
    max_frames: int,
) -> Optional[Dict]:
    from data.preprocessing.face_preprocessor import FacePreprocessor
    preprocessor = FacePreprocessor(image_size=image_size, fps=30, max_frames=max_frames)
    try:
        return preprocessor.preprocess(video_path)
    except Exception as exc:
        logger.debug("Face preprocessing failed for %s: %s", video_path, exc)
        return None


def preprocess_eeg(
    eeg_path: str,
    n_channels: int,
    sampling_rate: int,
) -> Optional[Dict]:
    from data.preprocessing.eeg_preprocessor import EEGPreprocessor
    preprocessor = EEGPreprocessor(
        sampling_rate=sampling_rate,
        n_channels=n_channels,
        segment_length=256,
    )
    try:
        raw = np.load(eeg_path)
        return preprocessor.preprocess(raw)
    except Exception as exc:
        logger.debug("EEG preprocessing failed for %s: %s", eeg_path, exc)
        return None


# ---------------------------------------------------------------------------
# Save utilities
# ---------------------------------------------------------------------------

def save_npz(output_path: str, data: Dict) -> None:
    """Save a dict of numpy arrays to a .npz file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            save_data[k] = v
        elif isinstance(v, (int, float)):
            save_data[k] = np.array(v)
    np.savez_compressed(output_path, **save_data)


# ---------------------------------------------------------------------------
# Dataset-specific subject discovery
# ---------------------------------------------------------------------------

def discover_subjects_daic_woz(data_root: str) -> List[Dict]:
    subjects = []
    for split_csv in ["train_split_Depression_AVEC2017.csv",
                      "dev_split_Depression_AVEC2017.csv",
                      "test_split_Depression_AVEC2017.csv"]:
        csv_path = os.path.join(data_root, split_csv)
        if not os.path.exists(csv_path):
            continue
        split_name = split_csv.split("_")[0]
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = str(row.get("Participant_ID", "")).strip()
                phq8 = float(row.get("PHQ8_Score", 0))
                participant_dir = os.path.join(data_root, f"{pid}_P")
                subjects.append({
                    "subject_id": pid,
                    "split": split_name,
                    "phq8_score": phq8,
                    "label": int(phq8 >= 10),
                    "audio_path": os.path.join(participant_dir, f"{pid}_AUDIO.wav"),
                    "transcript_path": os.path.join(participant_dir, f"{pid}_TRANSCRIPT.csv"),
                    "video_path": os.path.join(participant_dir, f"{pid}_VIDEO.mp4"),
                    "eeg_path": None,
                })
    return subjects


def discover_subjects_modma(data_root: str) -> List[Dict]:
    meta_path = os.path.join(data_root, "metadata.csv")
    if not os.path.exists(meta_path):
        return []
    import pandas as pd
    df = pd.read_csv(meta_path, dtype=str)
    df.columns = [c.lower().strip() for c in df.columns]
    subjects = []
    for _, row in df.iterrows():
        sid = str(row.get("subject_id", row.get("id", ""))).strip()
        label = int(float(row.get("label", 0)))
        subjects.append({
            "subject_id": sid,
            "split": str(row.get("split", "train")),
            "phq8_score": float(row.get("phq8_score", 15.0 if label else 5.0)),
            "label": label,
            "audio_path": os.path.join(data_root, "speech", f"{sid}.wav"),
            "transcript_path": None,
            "video_path": None,
            "eeg_path": os.path.join(data_root, "eeg", f"{sid}.npy"),
        })
    return subjects


def discover_subjects_pdch(data_root: str) -> List[Dict]:
    meta_path = os.path.join(data_root, "metadata.csv")
    if not os.path.exists(meta_path):
        return []
    import pandas as pd
    df = pd.read_csv(meta_path, dtype=str)
    df.columns = [c.lower().strip() for c in df.columns]
    subjects = []
    for _, row in df.iterrows():
        sid = str(row.get("subject_id", row.get("id", ""))).strip()
        label = int(float(row.get("label", 0)))
        subjects.append({
            "subject_id": sid,
            "split": str(row.get("split", "train")),
            "phq8_score": float(row.get("phq8_score", 15.0 if label else 5.0)),
            "label": label,
            "audio_path": os.path.join(data_root, "speech", f"{sid}.wav"),
            "transcript_path": os.path.join(data_root, "text", f"{sid}.txt"),
            "video_path": os.path.join(data_root, "video", f"{sid}.mp4"),
            "eeg_path": os.path.join(data_root, "eeg", f"{sid}.npy"),
        })
    return subjects


def process_subject(
    subject: Dict,
    output_dir: str,
    modalities: List[str],
    args: argparse.Namespace,
    overwrite: bool = False,
) -> Tuple[str, Dict[str, bool]]:
    """Process one subject and save results. Returns (subject_id, success_per_modality)."""
    sid = subject["subject_id"]
    subj_dir = os.path.join(output_dir, sid)
    os.makedirs(subj_dir, exist_ok=True)

    success: Dict[str, bool] = {}

    if "speech" in modalities:
        out_path = os.path.join(subj_dir, "speech.npz")
        if overwrite or not os.path.exists(out_path):
            audio_path = subject.get("audio_path", "")
            if audio_path and os.path.exists(audio_path):
                result = preprocess_speech(
                    audio_path, subject.get("transcript_path"),
                    args.sample_rate, args.max_audio_length
                )
                if result:
                    save_npz(out_path, result)
                    success["speech"] = True
                else:
                    success["speech"] = False
            else:
                success["speech"] = False
        else:
            success["speech"] = True

    if "text" in modalities:
        out_path = os.path.join(subj_dir, "text.npz")
        if overwrite or not os.path.exists(out_path):
            transcript_path = subject.get("transcript_path", "")
            result = preprocess_text(transcript_path)
            if result:
                save_npz(out_path, result)
                success["text"] = True
            else:
                success["text"] = False
        else:
            success["text"] = True

    if "face" in modalities:
        out_path = os.path.join(subj_dir, "face.npz")
        if overwrite or not os.path.exists(out_path):
            video_path = subject.get("video_path", "")
            if video_path and os.path.exists(video_path):
                result = preprocess_face(video_path, args.image_size, args.max_frames)
                if result:
                    save_npz(out_path, result)
                    success["face"] = True
                else:
                    success["face"] = False
            else:
                success["face"] = False
        else:
            success["face"] = True

    if "eeg" in modalities:
        out_path = os.path.join(subj_dir, "eeg.npz")
        if overwrite or not os.path.exists(out_path):
            eeg_path = subject.get("eeg_path", "")
            if eeg_path and os.path.exists(eeg_path):
                result = preprocess_eeg(eeg_path, args.eeg_n_channels, args.eeg_sampling_rate)
                if result:
                    save_npz(out_path, result)
                    success["eeg"] = True
                else:
                    success["eeg"] = False
            else:
                success["eeg"] = False
        else:
            success["eeg"] = True

    # Save subject metadata
    meta_path = os.path.join(subj_dir, "metadata.json")
    meta = {
        "subject_id": sid,
        "split": subject.get("split", "unknown"),
        "phq8_score": subject.get("phq8_score", 0.0),
        "label": subject.get("label", 0),
        "modalities_available": {k: v for k, v in success.items()},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return sid, success


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Dataset: %s | Modalities: %s", args.dataset, args.modalities)
    logger.info("Input: %s → Output: %s", args.data_root, args.output_dir)

    # Discover subjects
    if args.dataset == "daic_woz":
        subjects = discover_subjects_daic_woz(args.data_root)
    elif args.dataset == "modma":
        subjects = discover_subjects_modma(args.data_root)
    elif args.dataset == "pdch":
        subjects = discover_subjects_pdch(args.data_root)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.info("Found %d subjects.", len(subjects))

    if args.n_jobs > 1:
        # Parallel processing
        success_counts: Dict[str, int] = {m: 0 for m in args.modalities}
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = {
                executor.submit(
                    process_subject, subj, args.output_dir,
                    args.modalities, args, args.overwrite
                ): subj["subject_id"]
                for subj in subjects
            }
            with tqdm(total=len(subjects), desc="Processing") as pbar:
                for future in as_completed(futures):
                    try:
                        sid, success = future.result()
                        for m, ok in success.items():
                            if ok:
                                success_counts[m] += 1
                    except Exception as exc:
                        logger.warning("Error processing %s: %s", futures[future], exc)
                    pbar.update(1)
    else:
        # Sequential processing
        success_counts: Dict[str, int] = {m: 0 for m in args.modalities}
        for subj in tqdm(subjects, desc="Processing subjects"):
            sid, success = process_subject(
                subj, args.output_dir, args.modalities, args, args.overwrite
            )
            for m, ok in success.items():
                if ok:
                    success_counts[m] += 1

    logger.info("Preprocessing complete.")
    logger.info("Success counts (out of %d):", len(subjects))
    for m, count in success_counts.items():
        logger.info("  %-10s: %d / %d", m, count, len(subjects))


if __name__ == "__main__":
    main()
