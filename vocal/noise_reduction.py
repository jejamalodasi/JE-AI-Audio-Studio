from __future__ import annotations

import numpy as np
from scipy.signal import stft, istft


def spectral_gate(y: np.ndarray, sr: int, *, strength: float = 1.0) -> np.ndarray:
    """Conservative spectral gate. Designed to reduce steady background noise."""
    x = np.asarray(y, dtype=np.float32)
    if x.ndim == 1:
        return _gate_mono(x, sr, strength)
    return np.column_stack([_gate_mono(x[:, ch], sr, strength) for ch in range(x.shape[1])]).astype(np.float32)


def _gate_mono(x: np.ndarray, sr: int, strength: float) -> np.ndarray:
    if len(x) < 256:
        return x.copy()
    strength = float(np.clip(strength, 0.0, 2.0))
    nperseg = min(2048, max(512, 2 ** int(np.log2(min(len(x), 2048)))))
    noverlap = nperseg // 2
    f, t, z = stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary="zeros")
    mag = np.abs(z)
    noise_frames = max(1, min(mag.shape[1], int(0.12 * sr / max(1, nperseg - noverlap))))
    noise = np.median(mag[:, :noise_frames], axis=1, keepdims=True)
    threshold = noise * (1.8 + 2.2 * strength)
    ratio = np.clip(mag / np.maximum(threshold, 1e-9), 0.0, 1.0)
    mask = 0.12 + 0.88 * ratio
    clean = z * mask
    _, out = istft(clean, fs=sr, nperseg=nperseg, noverlap=noverlap, input_onesided=True)
    return np.asarray(out[: len(x)], dtype=np.float32)
