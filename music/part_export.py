from __future__ import annotations

from pathlib import Path
from typing import Any

from ai.vocal_to_music import VocalMusicConfig, generate_from_vocal


def build_music_part_midi_bundle(
    vocal_path: str,
    output_dir: str,
    bpm: float = 120.0,
    key: str = "C",
    scale: str = "major",
    bars: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate real MIDI parts for the timeline without synthesizing audio."""
    source = Path(vocal_path)
    if not source.exists():
        raise FileNotFoundError(str(source))

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    result = generate_from_vocal(
        str(source),
        output_midi=str(root / "melody.mid"),
        config=VocalMusicConfig(
            bpm=float(bpm),
            key=str(key or "C"),
            scale=str(scale or "major"),
            bars=max(1, int(bars)),
            seed=int(seed),
        ),
    )

    # Create one self-contained MIDI file per generated musical part.
    import mido

    def copy_track(track_name: str, track_index: int, filename: str) -> str:
        source_mid = mido.MidiFile(str(result["arrangement_midi_path"]))
        out = mido.MidiFile(ticks_per_beat=source_mid.ticks_per_beat)
        for index, track in enumerate(source_mid.tracks):
            name = ""
            for msg in track:
                if msg.type == "track_name":
                    name = msg.name
                    break
            if index == 0 or name == track_name:
                out.tracks.append(mido.MidiTrack(track))
        path = root / filename
        out.save(str(path))
        return str(path)

    parts = {
        "melody": str(result["midi_path"]),
        "chords": copy_track("Chords", 2, "chords.mid"),
        "bass": copy_track("Bass", 3, "bass.mid"),
        "drums": copy_track("Drums", 4, "drums.mid"),
        "rhythm": copy_track("Rhythm", 5, "rhythm.mid"),
        "arrangement": str(result["arrangement_midi_path"]),
    }

    return {
        "parts": parts,
        "bpm": float(bpm),
        "key": str(key),
        "scale": str(scale),
        "bars": max(1, int(bars)),
        "seed": int(seed),
    }
