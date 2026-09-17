from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def _validate_audio(y: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    if y is None:
        raise ValueError("No audio was provided.")
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError("The uploaded audio file is empty.")
    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if sr <= 0:
        raise ValueError("Invalid sample rate.")
    if y.ndim not in (1, 2):
        raise ValueError("Audio must be mono or stereo.")
    return y, int(sr)


def load_audio(path: str, mono: bool = False) -> Tuple[np.ndarray, int]:
    """Load audio safely. Uses soundfile first; librosa is used as a fallback for compressed formats."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError("Audio file could not be found.")

    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {suffix or 'unknown'}")

    try:
        y, sr = sf.read(path, always_2d=False, dtype="float32")
    except Exception:
        try:
            import librosa
            y, sr = librosa.load(path, sr=None, mono=mono)
            y = np.asarray(y, dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(
                "Could not decode this audio file. In Colab, install/enable FFmpeg and retry."
            ) from exc

    y, sr = _validate_audio(y, sr)
    if mono and y.ndim == 2:
        y = np.mean(y, axis=1, dtype=np.float32)
    return y, sr


def duration_seconds(y: np.ndarray, sr: int) -> float:
    y, sr = _validate_audio(y, sr)
    return float(y.shape[0] / sr)


def peak_dbfs(y: np.ndarray) -> float:
    peak = float(np.max(np.abs(y))) if np.size(y) else 0.0
    if peak <= 1e-12:
        return -120.0
    return float(20.0 * math.log10(min(peak, 1.0)))


def rms_dbfs(y: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64))) if np.size(y) else 0.0
    if rms <= 1e-12:
        return -120.0
    return float(20.0 * math.log10(min(rms, 1.0)))


def normalize(y: np.ndarray, target_peak: float = 0.98) -> np.ndarray:
    y, _ = _validate_audio(y, 1)
    if not 0 < target_peak <= 1:
        raise ValueError("target_peak must be between 0 and 1.")
    peak = float(np.max(np.abs(y)))
    if peak <= 1e-12:
        return y.copy()
    return np.clip(y * (target_peak / peak), -1.0, 1.0).astype(np.float32)


def trim_audio(y: np.ndarray, sr: int, start: float, end: Optional[float]) -> np.ndarray:
    y, sr = _validate_audio(y, sr)
    duration = duration_seconds(y, sr)
    start = max(0.0, float(start))
    end_value = duration if end is None else min(duration, float(end))
    if start >= end_value:
        raise ValueError("Trim start must be smaller than trim end.")
    a = int(round(start * sr))
    b = int(round(end_value * sr))
    return y[a:b].copy()


def save_wav(y: np.ndarray, sr: int, output_path: Optional[str] = None) -> str:
    y, sr = _validate_audio(y, sr)
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="je_ai_audio_")
        os.close(fd)
    sf.write(output_path, y, sr, subtype="PCM_16")
    return output_path
