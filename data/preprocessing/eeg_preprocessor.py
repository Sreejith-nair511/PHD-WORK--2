"""
EEG Preprocessor for DG-HMCF.

Applies bandpass filtering, artefact rejection, normalisation, and
segmentation to raw multi-channel EEG recordings.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt, iirnotch, sosfilt
except ImportError:
    butter = sosfiltfilt = iirnotch = sosfilt = None  # type: ignore


class EEGPreprocessor:
    """
    Preprocesses raw EEG data for depression detection.

    Parameters
    ----------
    sampling_rate : int
        Recording sampling rate in Hz.
    n_channels : int
        Number of EEG channels.
    segment_length : int
        Length of each segment in samples (default 256 ≈ 1 s at 256 Hz).
    bandpass_low : float
        Low-cut frequency for bandpass filter (Hz).
    bandpass_high : float
        High-cut frequency for bandpass filter (Hz).
    notch_freq : float
        Power-line notch frequency (Hz), 0 to disable.
    max_segments : int
        Maximum number of segments to return per recording.
    overlap : float
        Fractional overlap between consecutive segments [0, 1).
    """

    def __init__(
        self,
        sampling_rate: int = 256,
        n_channels: int = 64,
        segment_length: int = 256,
        bandpass_low: float = 0.5,
        bandpass_high: float = 50.0,
        notch_freq: float = 50.0,
        max_segments: int = 100,
        overlap: float = 0.5,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.segment_length = segment_length
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.max_segments = max_segments
        self.overlap = overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, eeg_data: np.ndarray) -> Dict:
        """
        Full preprocessing pipeline for one EEG recording.

        Parameters
        ----------
        eeg_data : np.ndarray
            Raw EEG data, shape (n_channels, n_timepoints) or
            (n_timepoints, n_channels) – auto-transposed.

        Returns
        -------
        dict with keys:
            ``segments``    – np.ndarray, shape (max_segments, n_channels, segment_length)
            ``segment_mask``– np.ndarray, shape (max_segments,), 1=real segment
            ``n_channels``  – int
            ``sampling_rate``– int
        """
        data = self._ensure_channels_first(eeg_data)

        # Trim/pad to requested number of channels
        data = self._adjust_channels(data)

        # Bandpass filter
        data = self._bandpass_filter(data, self.bandpass_low, self.bandpass_high)

        # Notch filter (power-line interference)
        if self.notch_freq > 0:
            data = self._notch_filter(data, self.notch_freq)

        # Artefact rejection: clip extreme values
        data = self._reject_artefacts(data)

        # Normalise per channel
        data = self._normalize(data)

        # Segment
        segments, n_real = self._segment(data)

        # Build mask
        segment_mask = np.zeros(self.max_segments, dtype=np.float32)
        segment_mask[:n_real] = 1.0

        return {
            "segments": segments.astype(np.float32),
            "segment_mask": segment_mask,
            "n_channels": self.n_channels,
            "sampling_rate": self.sampling_rate,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_channels_first(self, data: np.ndarray) -> np.ndarray:
        """Transpose to (n_channels, n_timepoints) if needed."""
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if data.ndim == 2:
            if data.shape[0] > data.shape[1]:
                # Likely (n_timepoints, n_channels) – transpose
                data = data.T
        return data

    def _adjust_channels(self, data: np.ndarray) -> np.ndarray:
        """Trim or zero-pad channel dimension to self.n_channels."""
        c = data.shape[0]
        if c == self.n_channels:
            return data
        elif c > self.n_channels:
            return data[: self.n_channels]
        else:
            pad = np.zeros((self.n_channels - c, data.shape[1]), dtype=data.dtype)
            return np.concatenate([data, pad], axis=0)

    def _bandpass_filter(
        self,
        data: np.ndarray,
        lowcut: float = 0.5,
        highcut: float = 50.0,
        order: int = 4,
    ) -> np.ndarray:
        """
        Apply a zero-phase Butterworth bandpass filter to each channel.
        """
        if butter is None or sosfiltfilt is None:
            return data
        nyq = self.sampling_rate / 2.0
        low = lowcut / nyq
        high = min(highcut / nyq, 0.999)
        if low >= high or low <= 0:
            return data
        try:
            sos = butter(order, [low, high], btype="band", output="sos")
            filtered = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered[ch] = sosfiltfilt(sos, data[ch])
            return filtered
        except Exception:
            return data

    def _notch_filter(
        self, data: np.ndarray, freq: float = 50.0, quality: float = 30.0
    ) -> np.ndarray:
        """Apply IIR notch filter to remove power-line noise."""
        if iirnotch is None or sosfilt is None:
            return data
        try:
            b, a = iirnotch(freq, quality, self.sampling_rate)
            filtered = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered[ch] = np.convolve(data[ch], b, mode="same")
            return filtered
        except Exception:
            return data

    def _reject_artefacts(self, data: np.ndarray, z_threshold: float = 5.0) -> np.ndarray:
        """
        Clip values more than ``z_threshold`` standard deviations from the
        channel mean.  A simple but effective artefact rejection strategy.
        """
        result = data.copy()
        for ch in range(result.shape[0]):
            ch_data = result[ch]
            mean = np.mean(ch_data)
            std = np.std(ch_data)
            if std < 1e-9:
                continue
            lo = mean - z_threshold * std
            hi = mean + z_threshold * std
            result[ch] = np.clip(ch_data, lo, hi)
        return result

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Z-score normalise each channel independently.
        """
        result = np.zeros_like(data, dtype=np.float32)
        for ch in range(data.shape[0]):
            ch_data = data[ch].astype(np.float32)
            mean = ch_data.mean()
            std = ch_data.std()
            if std < 1e-9:
                result[ch] = ch_data - mean
            else:
                result[ch] = (ch_data - mean) / std
        return result

    def _segment(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        Divide the continuous EEG into overlapping fixed-length segments.

        Returns
        -------
        segments   : np.ndarray, shape (max_segments, n_channels, segment_length)
        n_real     : int – number of real (non-padded) segments
        """
        n_channels, n_times = data.shape
        step = max(1, int(self.segment_length * (1.0 - self.overlap)))

        starts = list(range(0, n_times - self.segment_length + 1, step))
        segments_list: List[np.ndarray] = []
        for start in starts:
            seg = data[:, start: start + self.segment_length]
            segments_list.append(seg)
            if len(segments_list) >= self.max_segments:
                break

        n_real = len(segments_list)

        # Allocate output array (zero-padded)
        out = np.zeros(
            (self.max_segments, n_channels, self.segment_length), dtype=np.float32
        )
        for i, seg in enumerate(segments_list):
            out[i] = seg

        return out, n_real
