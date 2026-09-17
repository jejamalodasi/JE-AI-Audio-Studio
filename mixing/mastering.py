from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loudness import audio_stats


@dataclass(frozen=True)
class MasteringConfig:
    target_peak: float = 0.95
    compressor_threshold_db: float = -18.0
    compressor_ratio: float = 2.0
    makeup_db: float = 1.0
    saturation: float = 0.08


def _soft_compress(y: np.ndarray, threshold_db: float, ratio: float, makeup_db: float) -> np.ndarray:
    threshold = 10.0 ** (float(threshold_db) / 20.0)
    ratio = max(1.0, float(ratio))
    magnitude = np.abs(y)
    above = magnitude > threshold
    compressed = magnitude.copy()
    compressed[above] = threshold + (magnitude[above] - threshold) / ratio
    gain = 10.0 ** (float(makeup_db) / 20.0)
    return np.sign(y) * compressed * gain


def _soft_saturate(y: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0:
        return y
    drive = 1.0 + amount * 3.0
    normalized = np.tanh(y * drive)
    reference = np.tanh(drive)
    return normalized / max(reference, 1e-6)


def master_audio(y: np.ndarray, config: MasteringConfig | None = None) -> np.ndarray:
    """Apply a conservative, CPU-friendly master bus chain with a peak-safe limiter."""
    cfg = config or MasteringConfig()
    result = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    result = _soft_compress(result, cfg.compressor_threshold_db, cfg.compressor_ratio, cfg.makeup_db)
    result = _soft_saturate(result, cfg.saturation)

    peak = float(np.max(np.abs(result))) if result.size else 0.0
    target = float(np.clip(cfg.target_peak, 0.1, 0.999))
    if peak > target and peak > 1e-12:
        result = result * (target / peak)

    return np.clip(result, -1.0, 1.0).astype(np.float32)


def mastering_report(y: np.ndarray) -> dict[str, float]:
    return audio_stats(master_audio(y))
