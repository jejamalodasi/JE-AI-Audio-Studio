from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import find_peaks

from utils.audio_utils import _validate_audio, duration_seconds, peak_dbfs, rms_dbfs


def analyze_vocal(y: np.ndarray, sr: int) -> dict[str, Any]:
    """Return lightweight diagnostics used by the Vocal Fix pipeline."""
    y, sr = _validate_audio(y, sr)
    mono = np.mean(y, axis=1, dtype=np.float32) if y.ndim == 2 else y
    abs_y = np.abs(mono)
    peaks, _ = find_peaks(abs_y, height=max(0.02, float(np.percentile(abs_y, 75))), distance=max(1, int(sr * 0.08)))
    return {
        "duration_seconds": duration_seconds(y, sr),
        "sample_rate": int(sr),
        "channels": 1 if y.ndim == 1 else int(y.shape[1]),
        "rms_dbfs": rms_dbfs(y),
        "peak_dbfs": peak_dbfs(y),
        "crest_factor_db": peak_dbfs(y) - rms_dbfs(y),
        "estimated_events": int(len(peaks)),
        "silent": bool(np.max(abs_y) < 1e-5),
    }
