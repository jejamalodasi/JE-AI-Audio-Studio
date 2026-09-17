from __future__ import annotations

from dataclasses import dataclass

from .chord_generator import ChordConfig, generate_chords


@dataclass(frozen=True)
class BassConfig:
    bpm: float = 120.0
    bars: int = 4
    key: str = "C"
    scale: str = "major"
    octave: int = 2


def generate_bass(config: BassConfig | None = None) -> list[dict]:
    """Create a simple bassline following generated chord roots."""
    cfg = config or BassConfig()
    chords = generate_chords(ChordConfig(bpm=cfg.bpm, bars=cfg.bars, key=cfg.key, scale=cfg.scale))
    events = []
    for chord in chords:
        root = chord["notes"][0] - 24 + (12 * (int(cfg.octave) - 2))
        events.append({"start": chord["start"], "duration": chord["duration"] * 0.9, "note": root, "velocity": 96})
    return events
