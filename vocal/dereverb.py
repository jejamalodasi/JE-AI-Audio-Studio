from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DereverbConfig:
    strength: float = 0.35
    floor_db: float = -48.0
    fft_size: int = 2048
    hop_length: int = 512


def dereverb(y: np.ndarray, sr: int, config: DereverbConfig | None = None) -> np.ndarray:
    """Conservative spectral de-reverb foundation.

    This is a lightweight DSP stage. It estimates a local spectral floor and
    attenuates low-confidence diffuse energy rather than claiming full blind
    reverberation inversion. A future ML model can replace this backend.
    """
    import librosa

    cfg = config or DereverbConfig()
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")
    audio = np.asarray(y, dtype=np.float32)
    mono = audio if audio.ndim == 1 else np.mean(audio, axis=1)
    if mono.size == 0:
        return audio.copy()

    stft = librosa.stft(mono, n_fft=cfg.fft_size, hop_length=cfg.hop_length)
    mag = np.abs(stft)
    phase = np.angle(stft)
    if mag.shape[1] < 3:
        return audio.copy()

    # Reverb tends to create a diffuse, slowly changing spectral tail. A
    # frequency-wise lower percentile gives a conservative local floor.
    floor = np.percentile(mag, 20.0, axis=1, keepdims=True)
    floor = np.maximum(floor, 10.0 ** (cfg.floor_db / 20.0))
    excess = np.maximum(mag - floor, 0.0)
    ratio = excess / (mag + 1e-8)
    strength = float(np.clip(cfg.strength, 0.0, 1.0))
    gain = 1.0 - strength * (1.0 - ratio)
    cleaned = np.maximum(mag * gain, floor * 0.35)

    result = librosa.istft(cleaned * np.exp(1j * phase), hop_length=cfg.hop_length, length=mono.size)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if audio.ndim == 1:
        return result
    # Apply the same gain envelope to each channel to preserve stereo balance.
    channel_result = np.empty_like(audio)
    for ch in range(audio.shape[1]):
        channel_result[:, ch] = result + (audio[:, ch] - mono)
    return channel_result.astype(np.float32)
