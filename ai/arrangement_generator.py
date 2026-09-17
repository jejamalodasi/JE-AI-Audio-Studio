from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai.conditioned_music import ConditionedMusicConfig, condition_vocal_to_music
from music.midi_renderer import render_arrangement_midi
from utils.audio_utils import load_audio
from .vocal_to_melody import MelodyConfig, vocal_to_melody
from .basic_pitch_transcriber import transcribe_with_basic_pitch


@dataclass(frozen=True)
class AIConditionedArrangementConfig:
    """Settings for the vocal-conditioned MIDI arrangement pipeline."""

    bpm: float | None = None
    bars: int = 8
    seed: int = 42
    key: str | None = None
    scale: str | None = None
    melody_backend: str = "auto"
    min_frequency: float = 65.41
    max_frequency: float = 1046.50
    min_note_length_ms: float = 58.0
    onset_threshold: float = 0.50
    frame_threshold: float = 0.30
    density_floor: float = 0.28
    density_ceiling: float = 0.78
    swing: float = 0.0


def _melody_from_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "start": float(note.get("start", 0.0)),
            "duration": max(0.02, float(note.get("duration", 0.1))),
            "note": int(note.get("note", 60)),
            "velocity": float(note.get("velocity", 0.8)),
        }
        for note in notes
        if "note" in note
    ]


def _extract_melody(audio_path: str, y, sr: int, cfg: AIConditionedArrangementConfig) -> tuple[list[dict], str]:
    backend = str(cfg.melody_backend or "auto").strip().lower()
    if backend not in {"auto", "basic_pitch", "pyin"}:
        raise ValueError("melody_backend must be one of: auto, basic_pitch, pyin")

    if backend in {"auto", "basic_pitch"}:
        try:
            result = transcribe_with_basic_pitch(
                audio_path,
                minimum_frequency=float(cfg.min_frequency) if cfg.min_frequency > 0 else None,
                maximum_frequency=float(cfg.max_frequency) if cfg.max_frequency > 0 else None,
                minimum_note_length_ms=float(cfg.min_note_length_ms),
                onset_threshold=float(cfg.onset_threshold),
                frame_threshold=float(cfg.frame_threshold),
                midi_tempo=float(cfg.bpm or 120.0),
            )
            melody = _melody_from_notes(result["notes"])
            if melody:
                return melody, "spotify-basic-pitch"
        except (ImportError, RuntimeError):
            if backend == "basic_pitch":
                raise

    with tempfile.NamedTemporaryFile(suffix="_melody.mid", delete=False) as tmp:
        melody_path = tmp.name
    _, notes = vocal_to_melody(
        y,
        sr,
        output_path=melody_path,
        bpm=float(cfg.bpm or 120.0),
        config=MelodyConfig(fmin=float(cfg.min_frequency), fmax=float(cfg.max_frequency)),
    )
    return _melody_from_notes(notes), "librosa-pyin"


def generate_ai_conditioned_arrangement(
    audio_path: str,
    output_path: str | None = None,
    config: AIConditionedArrangementConfig | None = None,
) -> dict[str, Any]:
    """Generate a vocal-conditioned multi-track MIDI arrangement.

    This is a modular conditioning pipeline: vocal features determine tempo/key/
    density, while the current generators render synchronized accompaniment.
    Stronger learned arrangement models can replace the generator layer later.
    """
    if not audio_path:
        raise ValueError("audio_path is required")
    source = Path(audio_path)
    if not source.exists():
        raise FileNotFoundError(str(source))

    cfg = config or AIConditionedArrangementConfig()
    y, sr = load_audio(str(source))
    condition = condition_vocal_to_music(
        y,
        sr,
        ConditionedMusicConfig(
            bpm=cfg.bpm,
            bars=max(1, int(cfg.bars)),
            seed=int(cfg.seed),
            density_floor=float(cfg.density_floor),
            density_ceiling=float(cfg.density_ceiling),
        ),
    )

    bpm = float(condition["bpm"])
    key = str(cfg.key or condition["key"])
    scale = str(cfg.scale or condition["scale"])
    melody, melody_backend = _extract_melody(str(source), y, sr, cfg)

    if cfg.swing != 0.0:
        from music.rhythm_generator import RhythmConfig, generate_rhythm
        condition["rhythm"] = generate_rhythm(
            RhythmConfig(
                bpm=bpm,
                bars=max(1, int(cfg.bars)),
                density=float(condition["density"]),
                swing=float(cfg.swing),
                seed=int(cfg.seed),
            )
        )

    if output_path is None:
        arrangement = source.with_name(f"{source.stem}_ai_arrangement.mid")
    else:
        arrangement = Path(output_path)
        arrangement.parent.mkdir(parents=True, exist_ok=True)

    melody_output = arrangement.with_name(f"{arrangement.stem}_melody.mid")
    render_arrangement_midi(
        melody=melody,
        rhythm=condition["rhythm"],
        chords=condition["chords"],
        bass=condition["bass"],
        drums=condition["drums"],
        output_path=str(arrangement),
        bpm=bpm,
    )
    render_arrangement_midi(
        melody=melody,
        rhythm=None,
        chords=None,
        bass=None,
        drums=None,
        output_path=str(melody_output),
        bpm=bpm,
    )

    return {
        "arrangement_path": str(arrangement),
        "melody_path": str(melody_output),
        "melody": melody,
        "melody_backend": melody_backend,
        "bpm": bpm,
        "key": key,
        "scale": scale,
        "key_confidence": float(condition["key_confidence"]),
        "activity": float(condition["activity"]),
        "density": float(condition["density"]),
        "condition_backend": condition["backend"],
        "bars": max(1, int(cfg.bars)),
        "seed": int(cfg.seed),
    }
