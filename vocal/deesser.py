from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def deess(y: np.ndarray, sr: int, *, threshold: float = 0.22, reduction: float = 0.55) -> np.ndarray:
    """Simple broadband de-esser focused on the sibilance band."""
    x = np.asarray(y, dtype=np.float32)
    if x.ndim == 2:
        return np.column_stack([deess(x[:, i], sr, threshold=threshold, reduction=reduction) for i in range(x.shape[1])]).astype(np.float32)
    lo = min(4500.0, sr * 0.20)
    hi = min(10000.0, sr * 0.47)
    if hi <= lo or len(x) < 64:
        return x.copy()
    sos = butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    sib = sosfilt(sos, x)
    env = np.abs(sib)
    win = max(3, int(sr * 0.006))
    kernel = np.ones(win, dtype=np.float32) / win
    env = np.convolve(env, kernel, mode="same")
    amount = np.clip((env - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0) * np.clip(reduction, 0.0, 1.0)
    return np.clip(x - sib * amount, -1.0, 1.0).astype(np.float32)
