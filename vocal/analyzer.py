from __future__ import annotations

import numpy as np

from utils.audio_utils import rms_dbfs


def analyze_vocal(y: np.ndarray, sr: int) -> dict:
    """Lightweight analysis used by future Vocal Fix and AI modules.

    This intentionally avoids heavyweight ML dependencies in the base runtime.
    """
    if y.ndim == 2:
        mono = np.mean(y, axis=1, dtype=np.float32)
    else:
        mono = y.astype(np.float32, copy=False)

    duration = float(len(mono) / sr)
    zero_crossings = np.count_nonzero(np.diff(np.signbit(mono)))
    zcr = float(zero_crossings / max(1, len(mono)))

    return {
        "duration_seconds": duration,
        "sample_rate": int(sr),
        "rms_dbfs": rms_dbfs(mono),
        "zero_crossing_rate": zcr,
        "samples": int(len(mono)),
    }
