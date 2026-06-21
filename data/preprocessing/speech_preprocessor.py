"""
Speech Preprocessor for DG-HMCF.

Handles loading, resampling, interviewer-segment removal, normalization,
and behavioral feature extraction from raw audio files.
"""

import os
import json
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Dict, List, Tuple


class SpeechPreprocessor:
    """
    Preprocesses raw audio recordings for depression detection.

    Parameters
    ----------
    sample_rate : int
        Target sample rate in Hz (default 16 000).
    max_length : int
        Maximum waveform length in samples (default 160 000 = 10 s).
    remove_interviewer : bool
        Whether to strip interviewer turn segments using transcript timestamps.
    silence_threshold_db : float
        dB threshold below which a frame is considered silent (default -40).
    frame_length : int
        STFT frame length in samples (default 2048).
    hop_length : int
        STFT hop length in samples (default 512).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_length: int = 160000,
        remove_interviewer: bool = True,
        silence_threshold_db: float = -40.0,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> None:
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.remove_interviewer = remove_interviewer
        self.silence_threshold_db = silence_threshold_db
        self.frame_length = frame_length
        self.hop_length = hop_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(
        self,
        audio_path: str,
        transcript_path: Optional[str] = None,
    ) -> Dict:
        """
        Load and preprocess an audio file.

        Parameters
        ----------
        audio_path : str
            Path to the audio file (.wav / .mp3 / etc.).
        transcript_path : str, optional
            Path to a JSON/CSV transcript with speaker turn timestamps.
            Required when ``remove_interviewer=True``.

        Returns
        -------
        dict with keys:
            ``waveform``          – np.ndarray, shape (max_length,)
            ``attention_mask``    – np.ndarray, shape (max_length,), 1 = real, 0 = padded
            ``behavioral_features`` – np.ndarray, shape (6,)
            ``sample_rate``       – int
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        if self.remove_interviewer and transcript_path is not None:
            audio = self._remove_interviewer_segments(audio, transcript_path)

        # Normalise amplitude
        audio = self._normalize_audio(audio)

        # Pad / trim to fixed length
        waveform, attention_mask = self._pad_or_trim(audio)

        behavioral_features = self.extract_behavioral_features(audio, self.sample_rate)

        return {
            "waveform": waveform.astype(np.float32),
            "attention_mask": attention_mask.astype(np.float32),
            "behavioral_features": behavioral_features.astype(np.float32),
            "sample_rate": self.sample_rate,
        }

    def extract_behavioral_features(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """
        Compute a 6-dimensional behavioural feature vector.

        Returns
        -------
        np.ndarray, shape (6,)
            [speech_rate, pause_duration, silence_ratio,
             pitch_variance, energy_variance, response_latency]
        """
        speech_rate = self._compute_speech_rate(audio, sr)
        pause_duration = self._compute_pause_duration(audio, sr)
        silence_ratio = self._compute_silence_ratio(audio, sr)
        pitch_variance = self._compute_pitch_variance(audio, sr)
        energy_variance = self._compute_energy_variance(audio, sr)
        response_latency = self._compute_response_latency(audio, sr)

        features = np.array(
            [
                speech_rate,
                pause_duration,
                silence_ratio,
                pitch_variance,
                energy_variance,
                response_latency,
            ],
            dtype=np.float32,
        )

        # Replace NaN / Inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    # ------------------------------------------------------------------
    # Private helpers – feature computation
    # ------------------------------------------------------------------

    def _compute_speech_rate(self, audio: np.ndarray, sr: int) -> float:
        """
        Approximate syllables-per-second via energy-envelope peak counting.

        A rough syllable nucleus detector: count local maxima of the
        smoothed RMS energy envelope, then divide by signal duration.
        """
        duration = len(audio) / sr
        if duration < 0.1:
            return 0.0

        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]

        # Smooth with a moving average (window ~200 ms)
        window = max(1, int(0.2 * sr / self.hop_length))
        kernel = np.ones(window) / window
        smooth_rms = np.convolve(rms, kernel, mode="same")

        # Count peaks above median as syllable nuclei
        threshold = np.median(smooth_rms) * 1.2
        peaks = self._count_peaks(smooth_rms, threshold)

        return float(peaks / duration)

    def _compute_pause_duration(self, audio: np.ndarray, sr: int) -> float:
        """
        Compute mean pause length (in seconds) using silence detection.
        """
        silent_frames = self._get_silent_frames(audio, sr)
        if not silent_frames.any():
            return 0.0

        # Group consecutive silent frames into pause segments
        pauses = self._group_consecutive(np.where(silent_frames)[0])
        if not pauses:
            return 0.0

        frame_duration = self.hop_length / sr
        pause_durations = [len(p) * frame_duration for p in pauses]
        # Only count pauses > 200 ms (inter-word pauses)
        significant_pauses = [d for d in pause_durations if d > 0.2]
        return float(np.mean(significant_pauses)) if significant_pauses else 0.0

    def _compute_silence_ratio(self, audio: np.ndarray, sr: int) -> float:
        """
        Fraction of frames classified as silent.
        """
        silent_frames = self._get_silent_frames(audio, sr)
        if len(silent_frames) == 0:
            return 0.0
        return float(np.mean(silent_frames))

    def _compute_pitch_variance(self, audio: np.ndarray, sr: int) -> float:
        """
        Variance of the fundamental frequency (F0) over voiced segments.
        Uses librosa's pyin estimator.
        """
        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
            )
            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0
            voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
            if len(voiced_f0) < 2:
                return 0.0
            return float(np.var(voiced_f0))
        except Exception:
            return 0.0

    def _compute_energy_variance(self, audio: np.ndarray, sr: int) -> float:
        """
        Variance of short-time RMS energy.
        """
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]
        return float(np.var(rms))

    def _compute_response_latency(self, audio: np.ndarray, sr: int) -> float:
        """
        Time (in seconds) from the start of the recording to the first
        detected speech onset.
        """
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]
        threshold = np.max(rms) * 0.05  # 5% of peak energy
        onset_frames = np.where(rms > threshold)[0]
        if len(onset_frames) == 0:
            return float(len(audio) / sr)
        return float(onset_frames[0] * self.hop_length / sr)

    # ------------------------------------------------------------------
    # Private helpers – audio manipulation
    # ------------------------------------------------------------------

    def _remove_interviewer_segments(
        self,
        audio: np.ndarray,
        transcript_path: str,
    ) -> np.ndarray:
        """
        Zero-out or excise regions labelled as interviewer speech.

        Expects transcript JSON format:
        [{"speaker": "Ellie", "start": 1.2, "end": 3.4}, ...]
        """
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                turns = json.load(f)
        except Exception:
            return audio

        result = audio.copy()
        for turn in turns:
            speaker = turn.get("speaker", "").lower()
            if "ellie" in speaker or "interviewer" in speaker:
                start_sample = int(turn.get("start", 0) * self.sample_rate)
                end_sample = int(turn.get("end", 0) * self.sample_rate)
                start_sample = max(0, min(start_sample, len(result)))
                end_sample = max(0, min(end_sample, len(result)))
                result[start_sample:end_sample] = 0.0
        return result

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Peak-normalise audio to [-1, 1]."""
        peak = np.max(np.abs(audio))
        if peak > 1e-8:
            return audio / peak
        return audio

    def _pad_or_trim(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad with zeros or trim to ``self.max_length``.

        Returns
        -------
        waveform      : np.ndarray, shape (max_length,)
        attention_mask: np.ndarray, shape (max_length,) – 1 for real samples
        """
        length = len(audio)
        if length >= self.max_length:
            waveform = audio[: self.max_length]
            attention_mask = np.ones(self.max_length, dtype=np.float32)
        else:
            waveform = np.zeros(self.max_length, dtype=np.float32)
            waveform[:length] = audio
            attention_mask = np.zeros(self.max_length, dtype=np.float32)
            attention_mask[:length] = 1.0
        return waveform, attention_mask

    def _get_silent_frames(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Return boolean array indicating silent frames."""
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max(rms + 1e-9))
        return rms_db < self.silence_threshold_db

    @staticmethod
    def _count_peaks(signal: np.ndarray, threshold: float) -> int:
        """Count local maxima above threshold in 1-D signal."""
        count = 0
        for i in range(1, len(signal) - 1):
            if signal[i] > threshold and signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
                count += 1
        return count

    @staticmethod
    def _group_consecutive(indices: np.ndarray) -> List[List[int]]:
        """Group consecutive integers into sublists."""
        if len(indices) == 0:
            return []
        groups: List[List[int]] = []
        current = [indices[0]]
        for idx in indices[1:]:
            if idx == current[-1] + 1:
                current.append(idx)
            else:
                groups.append(current)
                current = [idx]
        groups.append(current)
        return groups
