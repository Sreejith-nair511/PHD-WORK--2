"""
Face / Video Preprocessor for DG-HMCF.

Extracts frame sequences from interview videos, detects and aligns faces,
and computes a set of behavioural features (smile frequency, gaze stability,
blink frequency, and proxy Action Units).
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


class FacePreprocessor:
    """
    Preprocesses video recordings of interview sessions.

    Parameters
    ----------
    image_size : int
        Target spatial resolution for face crops (square).
    fps : int
        Target frames-per-second to sample from the video.
    max_frames : int
        Maximum number of frames to retain (pads or trims).
    """

    def __init__(
        self,
        image_size: int = 224,
        fps: int = 30,
        max_frames: int = 300,
    ) -> None:
        self.image_size = image_size
        self.fps = fps
        self.max_frames = max_frames

        # Load Haar cascades for face / eye detection (shipped with OpenCV)
        self._face_cascade = None
        self._eye_cascade = None
        if cv2 is not None:
            cascade_face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade_eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
            if os.path.exists(cascade_face_path):
                self._face_cascade = cv2.CascadeClassifier(cascade_face_path)
            if os.path.exists(cascade_eye_path):
                self._eye_cascade = cv2.CascadeClassifier(cascade_eye_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, video_path: str) -> Dict:
        """
        Extract and normalise face frames from a video file.

        Parameters
        ----------
        video_path : str
            Path to the interview video (.mp4, .avi, …).

        Returns
        -------
        dict with keys:
            ``pixel_values``        – np.ndarray, shape (max_frames, 3, H, W)
            ``frame_mask``          – np.ndarray, shape (max_frames,), 1=real
            ``behavioral_features`` – np.ndarray, shape (7,)
        """
        frames = self._extract_frames(video_path, self.max_frames)
        if len(frames) == 0:
            # Return zeroed tensors when video cannot be loaded
            pixel_values = np.zeros(
                (self.max_frames, 3, self.image_size, self.image_size),
                dtype=np.float32,
            )
            frame_mask = np.zeros(self.max_frames, dtype=np.float32)
            behavioral_features = np.zeros(7, dtype=np.float32)
            return {
                "pixel_values": pixel_values,
                "frame_mask": frame_mask,
                "behavioral_features": behavioral_features,
            }

        processed = [self._preprocess_frame(f) for f in frames]
        n_real = len(processed)

        # Pad / trim to max_frames
        pixel_values = np.zeros(
            (self.max_frames, 3, self.image_size, self.image_size),
            dtype=np.float32,
        )
        frame_mask = np.zeros(self.max_frames, dtype=np.float32)
        for i, pf in enumerate(processed[: self.max_frames]):
            pixel_values[i] = pf
            frame_mask[i] = 1.0

        behavioral_features = self.extract_behavioral_features(frames)

        return {
            "pixel_values": pixel_values,
            "frame_mask": frame_mask,
            "behavioral_features": behavioral_features.astype(np.float32),
        }

    def extract_behavioral_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Compute 7-dimensional behavioural features from BGR frames.

        Returns
        -------
        np.ndarray, shape (7,)
            [smile_freq, gaze_stability, blink_freq, AU1, AU2, AU4, AU6]
        """
        smile_freq = self._estimate_smile_freq(frames)
        gaze_stability = self._estimate_gaze_stability(frames)
        blink_freq = self._estimate_blink_freq(frames)
        au1, au2, au4, au6 = self._estimate_action_units(frames)

        features = np.array(
            [smile_freq, gaze_stability, blink_freq, au1, au2, au4, au6],
            dtype=np.float32,
        )
        return np.nan_to_num(features, nan=0.0)

    # ------------------------------------------------------------------
    # Private helpers – frame extraction & processing
    # ------------------------------------------------------------------

    def _extract_frames(self, video_path: str, max_frames: int = 300) -> List[np.ndarray]:
        """
        Sample frames from video at the target FPS.

        Returns a list of BGR np.ndarrays.
        """
        if cv2 is None:
            return []
        if not os.path.exists(video_path):
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        frame_interval = max(1, int(round(video_fps / self.fps)))

        frames: List[np.ndarray] = []
        frame_idx = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                face_frame = self._detect_and_crop_face(frame)
                if face_frame is not None:
                    frames.append(face_frame)
            frame_idx += 1

        cap.release()
        return frames

    def _detect_and_crop_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the largest face and return a square crop.
        Falls back to centre-crop if no face is detected.
        """
        if self._face_cascade is None or frame is None:
            return self._centre_crop(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            return self._centre_crop(frame)

        # Pick the largest face by area
        faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
        x, y, w, h = faces[0]
        # Add 20% padding
        pad = int(0.2 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        face_crop = frame[y1:y2, x1:x2]
        return cv2.resize(face_crop, (self.image_size, self.image_size))

    def _centre_crop(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Return a centre-crop resized to ``image_size``."""
        if frame is None or frame.size == 0:
            return None
        if cv2 is None:
            return None
        h, w = frame.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = frame[y0: y0 + side, x0: x0 + side]
        return cv2.resize(crop, (self.image_size, self.image_size))

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert BGR frame to normalised CHW float32 tensor.

        Applies ImageNet mean/std normalisation compatible with ViT.
        """
        if cv2 is None:
            return np.zeros((3, self.image_size, self.image_size), dtype=np.float32)

        # Resize (should already be correct size, but ensure)
        frame = cv2.resize(frame, (self.image_size, self.image_size))
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # HWC -> float32 [0, 1]
        frame = frame.astype(np.float32) / 255.0
        # Normalise with ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frame = (frame - mean) / std
        # HWC -> CHW
        frame = frame.transpose(2, 0, 1)
        return frame

    # ------------------------------------------------------------------
    # Behavioural feature estimators
    # ------------------------------------------------------------------

    def _estimate_smile_freq(self, frames: List[np.ndarray]) -> float:
        """
        Approximate smile frequency (smiles per second) using the lower
        face region brightness ratio heuristic.

        A brighter lower half relative to upper half is a crude proxy for
        a smile (shows teeth).  For research-grade use, swap with
        AU12-based estimator.
        """
        if not frames:
            return 0.0
        smile_frames = 0
        for frame in frames:
            if frame is None:
                continue
            h = frame.shape[0]
            upper = frame[: h // 2]
            lower = frame[h // 2 :]
            if lower.mean() > upper.mean() * 1.05:
                smile_frames += 1
        duration_seconds = len(frames) / self.fps
        if duration_seconds < 0.1:
            return 0.0
        return float(smile_frames / duration_seconds)

    def _estimate_gaze_stability(self, frames: List[np.ndarray]) -> float:
        """
        Estimate gaze stability as 1 - normalised variance of the
        detected eye-region centroid across frames.
        Higher = more stable gaze.
        """
        if not frames or self._eye_cascade is None or cv2 is None:
            return 0.5  # neutral default
        centroids: List[Tuple[float, float]] = []
        for frame in frames[::3]:  # sample every 3rd frame for speed
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = self._eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(eyes) > 0:
                cx = float(np.mean([e[0] + e[2] / 2 for e in eyes]))
                cy = float(np.mean([e[1] + e[3] / 2 for e in eyes]))
                centroids.append((cx, cy))
        if len(centroids) < 2:
            return 0.5
        cx_arr = np.array([c[0] for c in centroids])
        cy_arr = np.array([c[1] for c in centroids])
        # Normalise variance by frame size
        variance = (np.var(cx_arr) + np.var(cy_arr)) / (self.image_size ** 2)
        stability = float(1.0 / (1.0 + variance))
        return float(np.clip(stability, 0.0, 1.0))

    def _estimate_blink_freq(self, frames: List[np.ndarray]) -> float:
        """
        Estimate blink frequency (blinks per second) by detecting
        rapid eye-area brightness drops between consecutive frames.
        """
        if not frames or cv2 is None:
            return 0.0
        eye_brightness: List[float] = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h = gray.shape[0]
            # Upper third of face typically contains eyes
            eye_region = gray[h // 4: h // 2]
            eye_brightness.append(float(eye_region.mean()))
        if len(eye_brightness) < 2:
            return 0.0
        brightness_arr = np.array(eye_brightness)
        # Detect drops > 15% relative to local mean
        local_mean = np.convolve(brightness_arr, np.ones(5) / 5, mode="same")
        drops = np.sum(brightness_arr < local_mean * 0.85)
        duration_seconds = len(frames) / self.fps
        return float(drops / duration_seconds) if duration_seconds > 0 else 0.0

    def _estimate_action_units(
        self, frames: List[np.ndarray]
    ) -> Tuple[float, float, float, float]:
        """
        Proxy estimates for Action Units AU1, AU2, AU4, AU6 using simple
        facial geometry heuristics.

        AU1 – Inner brow raise  (upper-face vertical gradient)
        AU2 – Outer brow raise  (forehead brightness)
        AU4 – Brow lowerer      (inverse of AU1+AU2)
        AU6 – Cheek raiser      (cheek-region brightness)

        Returns (AU1, AU2, AU4, AU6) each in [0, 1].
        """
        if not frames or cv2 is None:
            return 0.0, 0.0, 0.0, 0.0
        au1_vals, au2_vals, au4_vals, au6_vals = [], [], [], []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = gray.shape

            forehead = gray[: h // 5]
            brow = gray[h // 5: 2 * h // 5]
            cheek = gray[h // 2: 3 * h // 4]

            brow_mean = float(brow.mean())
            forehead_mean = float(forehead.mean())
            cheek_mean = float(cheek.mean())

            au1 = float(np.clip((brow_mean - forehead_mean + 128) / 255, 0, 1))
            au2 = float(np.clip(forehead_mean / 255, 0, 1))
            au4 = float(1.0 - (au1 + au2) / 2)
            au6 = float(np.clip(cheek_mean / 255, 0, 1))

            au1_vals.append(au1)
            au2_vals.append(au2)
            au4_vals.append(au4)
            au6_vals.append(au6)

        return (
            float(np.mean(au1_vals)),
            float(np.mean(au2_vals)),
            float(np.mean(au4_vals)),
            float(np.mean(au6_vals)),
        )
