from __future__ import annotations

import numpy as np


def vocal_fix(y: np.ndarray, sr: int, *, denoise: bool = False, normalize_output: bool = True) -> np.ndarray:
    """Safe placeholder pipeline for the future one-click Vocal Fix engine.

    The base version only performs conservative cleanup. Heavy AI processors will
    be plugged in here without changing the public app interface.
    """
    result = np.asarray(y, dtype=np.float32).copy()
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize_output:
        peak = float(np.max(np.abs(result))) if result.size else 0.0
        if peak > 1e-8:
            result *= min(0.98 / peak, 1.0)

    return np.clip(result, -1.0, 1.0).astype(np.float32)
