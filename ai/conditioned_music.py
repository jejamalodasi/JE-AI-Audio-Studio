from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from music.bass_generator import BassConfig, generate_bass
from music.chord_generator import ChordConfig, generate_chords
from music.drum_generator import DrumConfig, generate_drums
from music.rhythm_generator import RhythmConfig, generate_rhythm


@dataclass(frozen=True)
class ConditionedMusicConfig:
    """Configuration for turning detected vocal features into music controls."""

    bpm: float | None = None
    bars: int = 4
    seed: int = 42
    density_floor: float = 0.28
    density_ceiling: float = 0.78


_MAJOR_PROFILE = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
_MINOR_PROFILE = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)


def _estimate_key(chroma: np.ndarray) -> tuple[str, str, float]:
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    if chroma.size == 0:
        return "C", "major", 0.0
    profile_source = np.asarray(chroma, dtype=np.float32)
    profile_source = profile_source / max(1e-8, float(np.linalg.norm(profile_source)))
    best_key = 0
    best_scale = "major"
    best_score = -1.0
    for root in range(12):
        for scale_name, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            rotated = np.roll(profile, root)
            rotated = rotated / max(1e-8, float(np.linalg.norm(rotated)))
            score = float(np.dot(profile_source, rotated))
            if score > best_score:
                best_key, best_scale, best_score = root, scale_name, score
    return keys[best_key], best_scale, best_score


def condition_vocal_to_music(y: np.ndarray, sr: int, config: ConditionedMusicConfig | None = None) -> dict[str, Any]:
    """Extract compact vocal conditions and feed them into the existing music generators.

    This is an intermediate conditioning layer, not a learned end-to-end song model.
    """
    cfg = config or ConditionedMusicConfig()
    audio = np.asarray(y, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Audio must be a non-empty mono/stereo waveform.")
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")

    import librosa

    chroma = librosa.feature.chroma_cqt(y=audio, sr=int(sr), hop_length=512)
    chroma_mean = np.mean(chroma, axis=1)
    key, scale, key_confidence = _estimate_key(chroma_mean)

    tempo, _ = librosa.beat.beat_track(y=audio, sr=int(sr), trim=False)
    tempo = float(np.asarray(tempo).reshape(-1)[0]) if np.size(tempo) else 0.0
    if cfg.bpm is not None and float(cfg.bpm) > 0:
        tempo = float(cfg.bpm)
    if tempo <= 0:
        tempo = 120.0

    onset_env = librosa.onset.onset_strength(y=audio, sr=int(sr), hop_length=512)
    mean_onset = float(np.mean(onset_env)) if onset_env.size else 0.0
    max_onset = float(np.max(onset_env)) if onset_env.size else 1.0
    normalized_activity = float(np.clip(mean_onset / max(max_onset, 1e-6), 0.0, 1.0))
    density = float(np.clip(cfg.density_floor + (cfg.density_ceiling - cfg.density_floor) * normalized_activity, cfg.density_floor, cfg.density_ceiling))

    rhythm = generate_rhythm(RhythmConfig(bpm=tempo, bars=max(1, int(cfg.bars)), density=density, seed=int(cfg.seed)))
    chords = generate_chords(ChordConfig(bpm=tempo, bars=max(1, int(cfg.bars)), key=key, scale=scale))
    bass = generate_bass(BassConfig(bpm=tempo, bars=max(1, int(cfg.bars)), key=key, scale=scale))
    drums = generate_drums(DrumConfig(bpm=tempo, bars=max(1, int(cfg.bars)), density=density, seed=int(cfg.seed)))

    return {
        "bpm": tempo,
        "key": key,
        "scale": scale,
        "key_confidence": key_confidence,
        "activity": normalized_activity,
        "density": density,
        "rhythm": rhythm,
        "chords": chords,
        "bass": bass,
        "drums": drums,
        "backend": "librosa-conditioned-music-foundation",
    }
