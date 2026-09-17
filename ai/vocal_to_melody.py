from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class MelodyConfig:
    """Lightweight CPU-friendly vocal melody extraction settings."""

    fmin: float = 65.41   # C2
    fmax: float = 1046.50 # C6
    frame_length: int = 2048
    hop_length: int = 256
    voicing_threshold: float = 0.12
    min_note_seconds: float = 0.08
    midi_velocity: int = 90


def _clean_f0(f0: np.ndarray, threshold: float) -> np.ndarray:
    """Suppress very low-confidence/unvoiced frames."""
    f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
    f0[~np.isfinite(f0)] = 0.0
    # librosa.pyin represents unvoiced frames as NaN. Confidence is handled
    # by the caller; this helper keeps the output predictable.
    f0[f0 <= 0] = 0.0
    return f0


def _f0_to_midi(f0: np.ndarray) -> np.ndarray:
    midi = np.zeros_like(f0, dtype=np.float32)
    mask = f0 > 0
    midi[mask] = 69.0 + 12.0 * np.log2(f0[mask] / 440.0)
    return midi


def _write_midi(midi_notes: list[tuple[int, int, float, float]], path: str, bpm: float) -> str:
    """Write note tuples using mido, keeping MIDI dependency optional."""
    try:
        import mido
    except ImportError as exc:
        raise RuntimeError(
            "MIDI export needs the optional 'mido' package. Install requirements.txt first."
        ) from exc

    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="JE AI Melody"))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(float(bpm))))

    seconds_per_tick = (60.0 / float(bpm)) / mid.ticks_per_beat
    current_time = 0.0
    for note, velocity, start, duration in midi_notes:
        gap = max(0.0, start - current_time)
        track.append(mido.Message("note_on", note=note, velocity=velocity, time=round(gap / seconds_per_tick)))
        track.append(mido.Message("note_off", note=note, velocity=0, time=round(duration / seconds_per_tick)))
        current_time = start + duration

    mid.save(path)
    return path


def vocal_to_melody(
    y: np.ndarray,
    sr: int,
    output_path: Optional[str] = None,
    bpm: float = 120.0,
    config: Optional[MelodyConfig] = None,
) -> tuple[str, list[dict]]:
    """Extract a monophonic vocal melody and export it as MIDI.

    This first implementation intentionally uses librosa.pyin instead of a
    large neural model so it can run on CPU/Colab and gives the project a
    deterministic MIDI pipeline. Neural melody extraction can replace this
    backend later without changing the UI contract.

    Returns:
        (midi_path, notes) where notes contain pitch/start/duration metadata.
    """
    import librosa

    cfg = config or MelodyConfig()
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")
    audio = np.asarray(y, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Audio must be a non-empty mono or stereo waveform.")
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=cfg.fmin,
        fmax=min(cfg.fmax, sr / 2.0 - 1.0),
        sr=sr,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
        fill_na=np.nan,
    )
    f0 = np.asarray(f0, dtype=np.float32)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    voiced_prob = np.nan_to_num(np.asarray(voiced_prob, dtype=np.float32), nan=0.0)
    f0 = _clean_f0(f0, cfg.voicing_threshold)
    f0[~voiced_flag | (voiced_prob < cfg.voicing_threshold)] = 0.0

    midi_track = _f0_to_midi(f0)
    notes: list[tuple[int, int, float, float]] = []
    note_meta: list[dict] = []
    frame_seconds = cfg.hop_length / float(sr)

    active_note = None
    active_start = 0
    for idx, value in enumerate(midi_track):
        pitch = int(np.clip(np.rint(value), 0, 127)) if value > 0 else None
        if pitch != active_note:
            if active_note is not None:
                duration = (idx - active_start) * frame_seconds
                if duration >= cfg.min_note_seconds:
                    start = active_start * frame_seconds
                    notes.append((active_note, cfg.midi_velocity, start, duration))
                    note_meta.append({
                        "note": active_note,
                        "start": round(start, 4),
                        "duration": round(duration, 4),
                    })
            active_note = pitch
            active_start = idx

    if active_note is not None:
        duration = (len(midi_track) - active_start) * frame_seconds
        if duration >= cfg.min_note_seconds:
            start = active_start * frame_seconds
            notes.append((active_note, cfg.midi_velocity, start, duration))
            note_meta.append({
                "note": active_note,
                "start": round(start, 4),
                "duration": round(duration, 4),
            })

    if not notes:
        raise ValueError(
            "No confident melody was detected. Try a cleaner, mostly-monophonic vocal recording."
        )

    if output_path is None:
        output_path = str(Path.cwd() / "je_ai_melody.mid")
    output_path = str(Path(output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    _write_midi(notes, output_path, bpm=max(20.0, min(300.0, bpm)))
    return output_path, note_meta
