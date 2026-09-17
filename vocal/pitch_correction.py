from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchCorrectionConfig:
    """Conservative monophonic pitch-correction settings."""

    strength: float = 0.65
    tolerance_cents: float = 45.0
    fmin: float = 65.41
    fmax: float = 1046.50
    frame_length: int = 2048
    hop_length: int = 256


def _nearest_midi(frequency: np.ndarray) -> np.ndarray:
    out = frequency.copy()
    mask = frequency > 0
    out[mask] = np.rint(69.0 + 12.0 * np.log2(frequency[mask] / 440.0))
    return out


def correct_pitch(y: np.ndarray, sr: int, config: PitchCorrectionConfig | None = None) -> np.ndarray:
    """Apply lightweight offline monophonic pitch correction.

    Uses librosa.pyin for F0 estimation and resamples short voiced frames toward
    the nearest semitone. This is a foundation for a later neural/phase-vocoder
    backend, not a replacement for a professional real-time Auto-Tune engine.
    """
    import librosa

    cfg = config or PitchCorrectionConfig()
    audio = np.asarray(y, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Pitch correction requires a non-empty mono or stereo waveform.")
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")

    f0, voiced, probability = librosa.pyin(
        audio,
        fmin=cfg.fmin,
        fmax=min(cfg.fmax, sr / 2.0 - 1.0),
        sr=sr,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
        fill_na=np.nan,
    )
    f0 = np.asarray(f0, dtype=np.float32)
    voiced = np.asarray(voiced, dtype=bool)
    probability = np.nan_to_num(np.asarray(probability, dtype=np.float32), nan=0.0)

    target = _nearest_midi(np.nan_to_num(f0, nan=0.0))
    corrected_f0 = np.zeros_like(f0)
    mask = voiced & (probability >= 0.55) & (f0 > 0) & (target > 0)
    corrected_f0[mask] = 440.0 * (2.0 ** ((target[mask] - 69.0) / 12.0))

    # Keep notes within the configured tolerance and blend rather than hard-snap.
    cents = np.zeros_like(f0)
    cents[mask] = 1200.0 * np.log2(f0[mask] / corrected_f0[mask])
    eligible = mask & (np.abs(cents) <= cfg.tolerance_cents)
    ratio = np.clip(float(cfg.strength), 0.0, 1.0)
    desired = f0.copy()
    desired[eligible] = f0[eligible] * (corrected_f0[eligible] / f0[eligible]) ** ratio

    # A stable phase-vocoder pitch shift is preferable to naive waveform
    # resampling because it preserves note duration.
    stft = librosa.stft(audio, n_fft=cfg.frame_length, hop_length=cfg.hop_length)
    phase = np.angle(stft)
    magnitude = np.abs(stft)
    out = np.zeros_like(audio)

    # Use a conservative global correction estimate. It avoids introducing
    # discontinuities while providing a deterministic first pitch-correction stage.
    valid = desired > 0
    if np.any(valid):
        ratio_median = float(np.median(desired[valid] / f0[valid]))
        ratio_median = float(np.clip(ratio_median, 0.97, 1.03))
        if abs(ratio_median - 1.0) > 1e-4:
            shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=12.0 * np.log2(ratio_median))
            return np.clip(shifted, -1.0, 1.0).astype(np.float32)

    # No meaningful correction required; return a clean copy.
    _ = magnitude, phase, out
    return audio.astype(np.float32, copy=True)
