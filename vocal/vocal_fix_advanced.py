from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .breath_click import remove_breath_like_sections, remove_clicks
from .dereverb import DereverbConfig, dereverb
from .noise_reduction import spectral_gate
from .pitch_correction import PitchCorrectionConfig, correct_pitch
from .timing_correction import TimingCorrectionConfig, correct_timing


@dataclass(frozen=True)
class AdvancedVocalFixConfig:
    noise_reduction: float = 0.65
    dereverb: float = 0.30
    pitch_correction: float = 0.0
    timing_correction: float = 0.0
    breath_reduction: float = 0.20
    click_cleanup: bool = True


def advanced_vocal_fix(y: np.ndarray, sr: int, config: AdvancedVocalFixConfig | None = None) -> np.ndarray:
    """Run optional advanced vocal-cleanup stages in a conservative, safe order."""
    cfg = config or AdvancedVocalFixConfig()
    result = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    nr = float(np.clip(cfg.noise_reduction, 0.0, 2.0))
    if nr > 0:
        result = spectral_gate(result, sr, strength=nr)
    if cfg.click_cleanup:
        result = remove_clicks(result)
    if cfg.breath_reduction > 0:
        result = remove_breath_like_sections(
            result,
            sr,
            attenuation_db=4.0 + 8.0 * float(np.clip(cfg.breath_reduction, 0.0, 1.0)),
        )
    if cfg.dereverb > 0:
        result = dereverb(result, sr, DereverbConfig(strength=float(np.clip(cfg.dereverb, 0.0, 1.0))))
    if cfg.pitch_correction > 0:
        result = correct_pitch(result, sr, PitchCorrectionConfig(strength=float(np.clip(cfg.pitch_correction, 0.0, 1.0))))
    if cfg.timing_correction > 0:
        result = correct_timing(result, sr, TimingCorrectionConfig(strength=float(np.clip(cfg.timing_correction, 0.0, 1.0))))

    peak = float(np.max(np.abs(result))) if result.size else 0.0
    if peak > 0.98:
        result = result * (0.98 / peak)
    return np.clip(result, -1.0, 1.0).astype(np.float32)
