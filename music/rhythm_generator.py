from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RhythmConfig:
    bpm: float = 120.0
    bars: int = 4
    steps_per_bar: int = 16
    density: float = 0.45
    seed: int = 42


def generate_rhythm(config: RhythmConfig | None = None) -> dict:
    """Generate a deterministic 16-step rhythmic event grid.

    This is an algorithmic foundation for a later ML rhythm model. It returns
    beat timing metadata rather than rendered audio, making it easy to feed
    into drum/bass/arrangement renderers.
    """
    cfg = config or RhythmConfig()
    rng = np.random.default_rng(cfg.seed)
    steps = max(1, int(cfg.steps_per_bar))
    bars = max(1, int(cfg.bars))
    density = float(np.clip(cfg.density, 0.0, 1.0))
    step_seconds = 60.0 / max(float(cfg.bpm), 1.0) / 4.0

    events = []
    for bar in range(bars):
        for step in range(steps):
            accent = step % 4 == 0
            probability = min(1.0, density + (0.18 if accent else 0.0))
            if rng.random() < probability:
                events.append({
                    "bar": bar,
                    "step": step,
                    "time": (bar * steps + step) * step_seconds,
                    "velocity": 0.92 if accent else 0.65,
                })
    return {
        "bpm": float(cfg.bpm),
        "bars": bars,
        "steps_per_bar": steps,
        "step_seconds": step_seconds,
        "events": events,
    }
