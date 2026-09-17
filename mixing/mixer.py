from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from utils.audio_utils import load_audio, save_wav


def _to_stereo(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return np.column_stack((y, y)).astype(np.float32)
    if y.ndim == 2 and y.shape[1] == 1:
        return np.repeat(y, 2, axis=1).astype(np.float32)
    if y.ndim == 2 and y.shape[1] >= 2:
        return y[:, :2].astype(np.float32)
    raise ValueError("Audio must be mono or stereo.")


def _resample_stereo(y: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return y
    import librosa

    channels = [librosa.resample(y[:, ch], orig_sr=source_sr, target_sr=target_sr) for ch in range(y.shape[1])]
    return np.column_stack(channels).astype(np.float32)


def mix_audio_arrays(
    tracks: Iterable[np.ndarray],
    gains_db: Iterable[float] | None = None,
) -> np.ndarray:
    """Sum mono/stereo arrays into one stereo mix without intentional hard clipping."""
    arrays = [_to_stereo(track) for track in tracks]
    if not arrays:
        raise ValueError("At least one audio track is required.")

    gains = list(gains_db) if gains_db is not None else [0.0] * len(arrays)
    if len(gains) != len(arrays):
        raise ValueError("gains_db must contain one gain value per track.")

    length = max(track.shape[0] for track in arrays)
    mix = np.zeros((length, 2), dtype=np.float64)
    for track, gain_db in zip(arrays, gains):
        gain = 10.0 ** (float(gain_db) / 20.0)
        mix[: track.shape[0]] += track.astype(np.float64) * gain

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1.0:
        mix /= peak
    return np.clip(mix, -1.0, 1.0).astype(np.float32)


def mix_audio_files(
    paths: Iterable[str],
    gains_db: Iterable[float] | None = None,
    output_path: str | None = None,
) -> str:
    """Load and mix multiple audio files, resampling them to the first track's rate."""
    file_paths = [str(path) for path in paths if path]
    if not file_paths:
        raise ValueError("At least one audio file is required.")

    loaded = [load_audio(path) for path in file_paths]
    target_sr = loaded[0][1]
    arrays = [_resample_stereo(_to_stereo(y), sr, target_sr) for y, sr in loaded]
    mixed = mix_audio_arrays(arrays, gains_db=gains_db)
    return save_wav(mixed, target_sr, output_path=output_path)
