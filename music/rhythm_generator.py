from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RhythmConfig:
    bpm: float = 120.0
    bars: int = 4
    steps_per_beat: int = 4
    swing: float = 0.0
    density: float = 0.55
    seed: int = 42


def generate_rhythm(config: RhythmConfig | None = None) -> dict:
    """Generate a deterministic rhythmic event grid for downstream generators."""
    cfg = config or RhythmConfig()
    rng = np.random.default_rng(int(cfg.seed))
    bars = max(1, int(cfg.bars))
    spb = max(1, int(cfg.steps_per_beat))
    density = float(np.clip(cfg.density, 0.0, 1.0))
    swing = float(np.clip(cfg.swing, -0.5, 0.5))
    beat_seconds = 60.0 / max(float(cfg.bpm), 1.0)
    step_seconds = beat_seconds / spb
    events = []
    for step in range(bars * 4 * spb):
        pos = step % spb
        probability = min(1.0, density + (0.30 if pos == 0 else 0.08 if pos == spb // 2 and spb > 1 else 0.0))
        if rng.random() <= probability:
            time = step * step_seconds
            if spb > 1 and step % 2 == 1:
                time += swing * step_seconds
            events.append({
                "bar": step // (4 * spb) + 1,
                "step": step,
                "time": round(max(0.0, time), 6),
                "velocity": int(np.clip(78 + rng.normal(0, 10) + (18 if pos == 0 else 0), 1, 127)),
            })
    return {"bpm": float(cfg.bpm), "bars": bars, "steps_per_beat": spb, "step_seconds": step_seconds, "events": events}
