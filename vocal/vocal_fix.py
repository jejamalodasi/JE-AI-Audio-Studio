from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .deesser import deess
from .filters import gentle_compress, highpass
from .noise_reduction import spectral_gate


@dataclass(frozen=True)
class VocalFixConfig:
    highpass_hz: float = 70.0
    noise_reduction: float = 0.65
    deesser_threshold: float = 0.22
    deesser_reduction: float = 0.45
    compression_ratio: float = 2.0
    normalize_output: bool = True


def vocal_fix(y: np.ndarray, sr: int, config: VocalFixConfig | None = None) -> np.ndarray:
    """Run the safe first-generation one-click Vocal Fix DSP chain.

    This is deliberately conservative: it cleans and stabilizes recordings without
    pretending to perform ML pitch correction or source separation yet.
    """
    cfg = config or VocalFixConfig()
    result = np.asarray(y, dtype=np.float32).copy()
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    result = highpass(result, sr, cfg.highpass_hz)
    if cfg.noise_reduction > 0:
        result = spectral_gate(result, sr, strength=cfg.noise_reduction)
    result = deess(result, sr, threshold=cfg.deesser_threshold, reduction=cfg.deesser_reduction)
    result = gentle_compress(result, ratio=cfg.compression_ratio)

    if cfg.normalize_output:
        peak = float(np.max(np.abs(result))) if result.size else 0.0
        if peak > 1e-8:
            result *= min(0.98 / peak, 1.0)
    return np.clip(result, -1.0, 1.0).astype(np.float32)
