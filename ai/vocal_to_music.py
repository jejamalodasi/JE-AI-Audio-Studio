from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from music.bass_generator import BassConfig, generate_bass
from music.chord_generator import ChordConfig, generate_chords
from music.drum_generator import DrumConfig, generate_drums
from music.midi_renderer import render_arrangement_midi
from music.rhythm_generator import RhythmConfig, generate_rhythm
from .vocal_to_melody import MelodyConfig, vocal_to_melody


@dataclass(frozen=True)
class VocalMusicConfig:
    bpm: float = 120.0
    key: str = "C"
    scale: str = "major"
    bars: int = 4
    fmin: float = 65.41
    fmax: float = 1046.50
    seed: int = 42


def generate_from_vocal(
    audio_path: str,
    output_midi: str | None = None,
    config: VocalMusicConfig | None = None,
) -> dict[str, Any]:
    """Turn a vocal into melody plus a synchronized multi-track MIDI arrangement.

    The current stage uses F0-based melody extraction and deterministic
    accompaniment generators. It is a lightweight foundation for later learned
    rhythm/harmony/music-generation models.
    """
    cfg = config or VocalMusicConfig()
    if not audio_path:
        raise ValueError("A vocal input file is required.")
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    import tempfile
    from utils.audio_utils import load_audio

    y, sr = load_audio(str(path))
    if output_midi is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
        output_midi = tmp.name
        tmp.close()

    # Keep the melody MIDI generation for backwards compatibility.
    melody_midi_path, melody = vocal_to_melody(
        y,
        sr,
        output_path=output_midi,
        bpm=cfg.bpm,
        config=MelodyConfig(fmin=cfg.fmin, fmax=cfg.fmax),
    )

    rhythm = generate_rhythm(RhythmConfig(bpm=cfg.bpm, bars=cfg.bars, seed=cfg.seed))
    chords = generate_chords(ChordConfig(bpm=cfg.bpm, bars=cfg.bars, key=cfg.key, scale=cfg.scale))
    bass = generate_bass(BassConfig(bpm=cfg.bpm, bars=cfg.bars, key=cfg.key, scale=cfg.scale))
    drums = generate_drums(DrumConfig(bpm=cfg.bpm, bars=cfg.bars, seed=cfg.seed))

    arrangement_path = Path(output_midi).with_name(Path(output_midi).stem + "_arrangement.mid")
    render_arrangement_midi(
        melody=melody,
        rhythm=rhythm,
        chords=chords,
        bass=bass,
        drums=drums,
        output_path=str(arrangement_path),
        bpm=cfg.bpm,
    )

    return {
        "midi_path": melody_midi_path,
        "arrangement_midi_path": str(arrangement_path),
        "sample_rate": sr,
        "melody": melody,
        "rhythm": rhythm,
        "chords": chords,
        "bass": bass,
        "drums": drums,
        "config": cfg,
    }
