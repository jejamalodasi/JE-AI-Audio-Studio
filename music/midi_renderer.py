from __future__ import annotations

from pathlib import Path
from typing import Iterable

import mido

from .gm_drum_map import resolve_drum_note

NOTE_MIN = 0
NOTE_MAX = 127


def _clamp_note(note: int | float) -> int:
    return max(NOTE_MIN, min(NOTE_MAX, int(round(note))))


def _clamp_velocity(value: int | float) -> int:
    value = float(value)
    if 0.0 <= value <= 1.0:
        value *= 127.0
    return max(1, min(127, int(round(value))))


def _add_note_events(events: list[tuple[float, str, int, int]], start: float, duration: float, note: int | float, velocity: int | float = 90) -> None:
    start = max(0.0, float(start))
    duration = max(0.01, float(duration))
    n = _clamp_note(note)
    v = _clamp_velocity(velocity)
    events.append((start, "on", n, v))
    events.append((start + duration, "off", n, 0))


def _seconds_to_ticks(seconds: float, bpm: float, ticks_per_beat: int) -> int:
    beats = max(0.0, float(seconds)) * max(1.0, float(bpm)) / 60.0
    return int(round(beats * ticks_per_beat))


def render_arrangement_midi(
    melody: Iterable[dict] | None,
    rhythm: dict | None,
    chords: Iterable[dict] | None,
    bass: Iterable[dict] | None,
    drums: Iterable[dict] | None,
    output_path: str,
    bpm: float = 120.0,
    ticks_per_beat: int = 480,
) -> str:
    """Render generated metadata into one multi-track MIDI arrangement.

    Percussion uses the common General MIDI drum-note map on channel 10.
    The renderer remains lightweight and does not synthesize audio.
    """
    if not output_path:
        raise ValueError("output_path is required")

    bpm = max(1.0, float(bpm))
    mid = mido.MidiFile(ticks_per_beat=int(ticks_per_beat))

    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("track_name", name="JE AI Tempo", time=0))
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    mid.tracks.append(tempo_track)

    def add_track(name: str, channel: int, events: list[tuple[float, str, int, int]]) -> None:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        events.sort(key=lambda item: (item[0], 0 if item[1] == "off" else 1, item[2]))
        previous_ticks = 0
        for seconds, kind, note, velocity in events:
            absolute_ticks = _seconds_to_ticks(seconds, bpm, mid.ticks_per_beat)
            delta = max(0, absolute_ticks - previous_ticks)
            if kind == "on":
                track.append(mido.Message("note_on", channel=channel, note=note, velocity=velocity, time=delta))
            else:
                track.append(mido.Message("note_off", channel=channel, note=note, velocity=0, time=delta))
            previous_ticks = absolute_ticks
        mid.tracks.append(track)

    melody_events: list[tuple[float, str, int, int]] = []
    for item in melody or []:
        if "start" in item and "note" in item:
            _add_note_events(melody_events, item["start"], item.get("duration", 0.25), item["note"], item.get("velocity", 100))
        elif "time" in item and "midi" in item:
            _add_note_events(melody_events, item["time"], item.get("duration", 0.25), item["midi"], item.get("velocity", 100))

    chord_events: list[tuple[float, str, int, int]] = []
    for item in chords or []:
        start = item.get("start", item.get("time", 0.0))
        duration = item.get("duration", 1.0)
        for note in item.get("notes", []):
            _add_note_events(chord_events, start, duration, note, item.get("velocity", 68))

    bass_events: list[tuple[float, str, int, int]] = []
    for item in bass or []:
        _add_note_events(bass_events, item.get("start", item.get("time", 0.0)), item.get("duration", 0.5), item.get("note", 36), item.get("velocity", 96))

    drum_events: list[tuple[float, str, int, int]] = []
    for item in drums or []:
        raw_note = item.get("drum", item.get("instrument", item.get("note", item.get("midi", 36))))
        _add_note_events(
            drum_events,
            item.get("start", item.get("time", 0.0)),
            item.get("duration", 0.08),
            resolve_drum_note(raw_note),
            item.get("velocity", 100),
        )

    rhythm_events: list[tuple[float, str, int, int]] = []
    if rhythm:
        for item in rhythm.get("events", []):
            _add_note_events(
                rhythm_events,
                item.get("time", 0.0),
                rhythm.get("step_seconds", 0.125) * 0.75,
                42,
                item.get("velocity", 96),
            )

    add_track("Melody", 0, melody_events)
    add_track("Chords", 1, chord_events)
    add_track("Bass", 2, bass_events)
    add_track("Drums", 9, drum_events)
    add_track("Rhythm", 3, rhythm_events)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))
    return str(path)
