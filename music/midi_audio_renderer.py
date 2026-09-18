from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
import soundfile as sf


def _events_to_notes(mid: mido.MidiFile) -> list[tuple[float, float, int, int, int]]:
    notes: list[tuple[float, float, int, int, int]] = []
    for track in mid.tracks:
        absolute = 0
        active: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for message in track:
            absolute += int(message.time)
            if message.type not in {"note_on", "note_off"}:
                continue
            note = int(getattr(message, "note", 60))
            channel = int(getattr(message, "channel", 0))
            velocity = int(getattr(message, "velocity", 0))
            key = (channel, note)
            if message.type == "note_on" and velocity > 0:
                active.setdefault(key, []).append((absolute, velocity))
                continue
            queued = active.get(key)
            if not queued:
                continue
            start_tick, start_velocity = queued.pop(0)
            if absolute > start_tick:
                start = start_tick / mid.ticks_per_beat
                duration = (absolute - start_tick) / mid.ticks_per_beat
                notes.append((start, duration, note, start_velocity, channel))
            if not queued:
                active.pop(key, None)
    return notes


def render_midi_to_audio(
    midi_path: str,
    output_path: str,
    bpm: float = 120.0,
    sample_rate: int = 44100,
    duration_limit_seconds: float = 1800.0,
) -> str:
    """Render MIDI clips to lightweight, deterministic audio without external synth binaries.

    This is a real audio render, intentionally using a small built-in synth:
    melodic notes use additive sine harmonics; channel 10 percussion uses short noise/click tones.
    It is a renderer foundation, not a sample-library replacement.
    """
    if not midi_path:
        raise ValueError("midi_path is required")
    bpm = max(1.0, float(bpm))
    sr = int(sample_rate)
    mid = mido.MidiFile(str(midi_path))
    notes = _events_to_notes(mid)
    if not notes:
        return sf.write(str(output_path), np.zeros((1, 2), dtype=np.float32), sr) or str(output_path)

    max_beat = max(start + duration for start, duration, *_ in notes)
    total_seconds = min(duration_limit_seconds, max_beat * 60.0 / bpm + 0.25)
    total_samples = max(1, int(np.ceil(total_seconds * sr)))
    mono = np.zeros(total_samples, dtype=np.float64)

    for start_beat, duration_beat, note, velocity, channel in notes:
        start_sec = start_beat * 60.0 / bpm
        dur_sec = min(duration_beat * 60.0 / bpm, total_seconds - start_sec)
        if dur_sec <= 0 or start_sec >= total_seconds:
            continue
        start_i = max(0, int(round(start_sec * sr)))
        count = min(total_samples - start_i, max(1, int(round(dur_sec * sr))))
        t = np.arange(count, dtype=np.float64) / sr
        amp = min(1.0, max(0.02, velocity / 127.0)) * (0.22 if channel == 9 else 0.14)

        if channel == 9:
            rng = np.random.default_rng(note * 7919 + start_i)
            noise = rng.standard_normal(count)
            decay = np.exp(-t * (28.0 if note in {42, 46} else 18.0))
            tone = np.sin(2 * np.pi * (110.0 + (note % 12) * 7.0) * t)
            wave = (0.78 * noise + 0.22 * tone) * decay
        else:
            freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
            wave = (
                np.sin(2 * np.pi * freq * t)
                + 0.32 * np.sin(2 * np.pi * freq * 2.0 * t)
                + 0.14 * np.sin(2 * np.pi * freq * 3.0 * t)
            )
            attack = min(0.015, dur_sec * 0.25)
            release = min(0.08, dur_sec * 0.35)
            envelope = np.ones(count)
            if attack > 0:
                envelope *= np.minimum(1.0, t / attack)
            if release > 0:
                envelope *= np.minimum(1.0, np.maximum(0.0, (dur_sec - t) / release))
            wave *= envelope

        mono[start_i:start_i + count] += amp * wave

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 0.95:
        mono *= 0.95 / peak
    stereo = np.column_stack((mono, mono)).astype(np.float32)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), stereo, sr)
    return str(output_path)
