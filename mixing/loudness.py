from __future__ import annotations

import math

import numpy as np


def audio_stats(y: np.ndarray) -> dict[str, float]:
    """Return lightweight peak/RMS/dynamic-range statistics for an audio array."""
    data = np.asarray(y, dtype=np.float32)
    if data.size == 0:
        return {"peak_dbfs": -120.0, "rms_dbfs": -120.0, "crest_db": 0.0}
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(np.square(data), dtype=np.float64)))
    peak_db = -120.0 if peak <= 1e-12 else 20.0 * math.log10(min(peak, 1.0))
    rms_db = -120.0 if rms <= 1e-12 else 20.0 * math.log10(min(rms, 1.0))
    return {"peak_dbfs": peak_db, "rms_dbfs": rms_db, "crest_db": peak_db - rms_db}
