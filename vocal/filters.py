from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def highpass(y: np.ndarray, sr: int, cutoff: float = 70.0) -> np.ndarray:
    """Remove DC/low rumble while preserving the vocal body."""
    if y.ndim == 2:
        return np.column_stack([highpass(y[:, i], sr, cutoff) for i in range(y.shape[1])]).astype(np.float32)
    cutoff = float(np.clip(cutoff, 20.0, sr * 0.45))
    sos = butter(3, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, y).astype(np.float32)


def gentle_compress(y: np.ndarray, threshold_db: float = -18.0, ratio: float = 2.0, makeup_db: float = 1.5) -> np.ndarray:
    """Lightweight peak-domain compressor; intentionally conservative."""
    x = np.asarray(y, dtype=np.float32)
    mag = np.abs(x)
    db = 20.0 * np.log10(np.maximum(mag, 1e-8))
    over = np.maximum(db - threshold_db, 0.0)
    gain_db = -over * (1.0 - 1.0 / max(1.0, ratio)) + makeup_db
    gain = 10.0 ** (gain_db / 20.0)
    return np.clip(x * gain, -1.0, 1.0).astype(np.float32)
