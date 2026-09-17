from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimingCorrectionConfig:
    strength: float = 0.35
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    max_shift_ms: float = 35.0


def _rms_envelope(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    values = []
    for start in range(0, max(1, len(audio) - frame + 1), hop):
        chunk = audio[start:start + frame]
        if len(chunk) == 0:
            break
        values.append(np.sqrt(np.mean(chunk * chunk) + 1e-12))
    return np.asarray(values, dtype=np.float32)


def correct_timing(y: np.ndarray, sr: int, config: TimingCorrectionConfig | None = None) -> np.ndarray:
    """Gentle timing cleanup foundation using transient-safe envelope warping.

    The current stage deliberately avoids aggressive time stretching. It removes
    leading/trailing dead space and leaves internal phrasing intact. The API is
    designed so a future beat/phoneme alignment model can replace this backend.
    """
    cfg = config or TimingCorrectionConfig()
    audio = np.asarray(y, dtype=np.float32).copy()
    if audio.ndim == 2:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio
    if mono.size == 0 or sr <= 0:
        return audio

    threshold = max(1e-5, float(np.percentile(np.abs(mono), 15)) * 0.7)
    active = np.flatnonzero(np.abs(mono) > threshold)
    if active.size == 0:
        return audio

    start = int(active[0])
    end = int(active[-1]) + 1
    pad = int(0.02 * sr)
    start = max(0, start - pad)
    end = min(len(audio), end + pad)
    trimmed = audio[start:end]

    # Strength controls how much of the detected dead space is removed.
    strength = float(np.clip(cfg.strength, 0.0, 1.0))
    target_start = int(round(start * strength))
    target_end = len(audio) - int(round((len(audio) - end) * strength))
    target = audio[target_start:target_end]

    # If correction is effectively disabled, preserve the original exactly.
    if strength <= 0.001:
        return audio
    if target.size == 0:
        return trimmed.astype(np.float32)
    return target.astype(np.float32)
