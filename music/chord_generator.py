from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChordConfig:
    bpm: float = 120.0
    bars: int = 4
    key: str = "C"
    scale: str = "major"
    progression: tuple[int, ...] = (1, 5, 6, 4)


_MAJOR = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
_MINOR = (0, 2, 3, 5, 7, 8, 10)
_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)


def generate_chords(config: ChordConfig | None = None) -> list[dict]:
    """Generate a MIDI-note chord progression as timing metadata."""
    cfg = config or ChordConfig()
    root = _MAJOR.get(cfg.key.upper(), 0)
    scale = _MINOR if cfg.scale.lower() == "minor" else _MAJOR_SCALE
    progression = cfg.progression or (1, 5, 6, 4)
    beat = 60.0 / max(1.0, float(cfg.bpm))
    chords = []
    for bar in range(max(1, int(cfg.bars))):
        degree = int(progression[bar % len(progression)])
        idx = (degree - 1) % 7
        third = (idx + 2) % 7
        fifth = (idx + 4) % 7
        notes = [60 + root + scale[idx], 60 + root + scale[third], 60 + root + scale[fifth]]
        notes = [((n - 60) % 12) + 60 for n in notes]
        chords.append({"bar": bar + 1, "start": bar * 4 * beat, "duration": 4 * beat, "degree": degree, "notes": notes})
    return chords
