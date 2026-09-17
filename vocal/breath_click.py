from __future__ import annotations

import numpy as np


def remove_clicks(y: np.ndarray, threshold: float = 0.92, radius: int = 16) -> np.ndarray:
    """Reduce isolated sample spikes without aggressively changing the vocal."""
    audio = np.asarray(y, dtype=np.float32).copy()
    if audio.ndim == 2:
        for ch in range(audio.shape[1]):
            audio[:, ch] = remove_clicks(audio[:, ch], threshold, radius)
        return audio
    if audio.ndim != 1 or audio.size < 3:
        return audio

    abs_audio = np.abs(audio)
    candidates = np.where(abs_audio >= threshold)[0]
    for idx in candidates:
        left = max(0, idx - radius)
        right = min(audio.size, idx + radius + 1)
        local = audio[left:right]
        if local.size < 3:
            continue
        median = float(np.median(local))
        deviation = abs(float(audio[idx]) - median)
        neighborhood = np.delete(local, idx - left)
        scale = float(np.median(np.abs(neighborhood - np.median(neighborhood)))) + 1e-5
        if deviation > 12.0 * scale:
            audio[idx] = median
    return audio


def remove_breath_like_sections(
    y: np.ndarray,
    sr: int,
    threshold_db: float = -42.0,
    min_duration: float = 0.12,
    attenuation_db: float = 8.0,
) -> np.ndarray:
    """Gently attenuate sustained low-level sections; preserves silence boundaries.

    This is intentionally conservative. Breath detection is difficult without a
    trained classifier, so this function only attenuates clearly quiet sections.
    """
    audio = np.asarray(y, dtype=np.float32).copy()
    if audio.ndim == 2:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio
    if mono.size == 0 or sr <= 0:
        return audio

    frame = max(256, int(round(min_duration * sr)))
    hop = max(128, frame // 2)
    threshold = 10.0 ** (threshold_db / 20.0)
    gain = 10.0 ** (-abs(attenuation_db) / 20.0)

    quiet = np.zeros(mono.size, dtype=bool)
    for start in range(0, mono.size, hop):
        end = min(mono.size, start + frame)
        if np.sqrt(np.mean(mono[start:end] ** 2) + 1e-12) < threshold:
            quiet[start:end] = True

    result = audio.copy()
    if result.ndim == 1:
        result[quiet] *= gain
    else:
        result[quiet, :] *= gain
    return result
