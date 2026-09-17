from __future__ import annotations

from dataclasses import dataclass

from .rhythm_generator import RhythmConfig, generate_rhythm


@dataclass(frozen=True)
class DrumConfig:
    bpm: float = 120.0
    bars: int = 4
    density: float = 0.55
    seed: int = 42


def generate_drums(config: DrumConfig | None = None) -> list[dict]:
    """Map a rhythm grid onto a compact kick/snare/hat drum pattern."""
    cfg = config or DrumConfig()
    grid = generate_rhythm(RhythmConfig(bpm=cfg.bpm, bars=cfg.bars, density=cfg.density, seed=cfg.seed))
    events = []
    for item in grid["events"]:
        pos = item["step"] % 16
        if pos in (0, 8):
            drum = "kick"
        elif pos in (4, 12):
            drum = "snare"
        else:
            drum = "hat"
        events.append({"time": item["time"], "drum": drum, "velocity": item["velocity"]})
    return events
