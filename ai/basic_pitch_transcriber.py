from __future__ import annotations

from pathlib import Path
from typing import Any


def transcribe_with_basic_pitch(
    audio_path: str,
    output_dir: str | None = None,
    minimum_frequency: float | None = None,
    maximum_frequency: float | None = None,
    minimum_note_length_ms: float = 58.0,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    midi_tempo: float = 120.0,
) -> dict[str, Any]:
    """Run Spotify Basic Pitch as an optional neural audio-to-MIDI backend.

    The dependency is imported lazily so the normal CPU/DSP installation stays
    lightweight. Basic Pitch returns a MIDI object with pitch-bend information
    and note events; this wrapper writes the MIDI and exposes compact metadata.
    """
    if not audio_path:
        raise ValueError("audio_path is required")
    source = Path(audio_path)
    if not source.exists():
        raise FileNotFoundError(str(source))

    try:
        from basic_pitch.inference import predict
    except ImportError as exc:
        raise RuntimeError(
            "Basic Pitch is not installed. Install the optional dependency with: "
            "pip install basic-pitch"
        ) from exc

    destination = Path(output_dir) if output_dir else source.parent / "basic_pitch_output"
    destination.mkdir(parents=True, exist_ok=True)

    _, midi_data, note_events = predict(
        str(source),
        onset_threshold=float(onset_threshold),
        frame_threshold=float(frame_threshold),
        minimum_note_length=float(minimum_note_length_ms),
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        midi_tempo=float(midi_tempo),
    )

    midi_path = destination / f"{source.stem}_basic_pitch.mid"
    midi_data.write(str(midi_path))

    notes = [
        {
            "start": float(start),
            "end": float(end),
            "duration": max(0.0, float(end) - float(start)),
            "note": int(pitch),
            "velocity": float(amplitude),
            "pitch_bend": pitch_bend,
        }
        for start, end, pitch, amplitude, pitch_bend in note_events
    ]

    return {
        "midi_path": str(midi_path),
        "notes": notes,
        "note_count": len(notes),
        "backend": "spotify-basic-pitch",
    }
