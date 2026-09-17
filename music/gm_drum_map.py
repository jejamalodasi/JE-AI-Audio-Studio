"""Common General MIDI percussion note mappings."""

GM_DRUM_NOTES = {
    "kick": 36,
    "bass_drum": 36,
    "kick_drum": 36,
    "snare": 38,
    "snare_drum": 38,
    "clap": 39,
    "handclap": 39,
    "closed_hat": 42,
    "closed_hihat": 42,
    "closed_hi_hat": 42,
    "hihat": 42,
    "hi_hat": 42,
    "pedal_hat": 44,
    "pedal_hihat": 44,
    "open_hat": 46,
    "open_hihat": 46,
    "open_hi_hat": 46,
    "low_tom": 45,
    "tom_low": 45,
    "mid_tom": 47,
    "tom_mid": 47,
    "high_tom": 50,
    "tom_high": 50,
    "crash": 49,
    "crash_cymbal": 49,
    "ride": 51,
    "ride_cymbal": 51,
}


def resolve_drum_note(value) -> int:
    """Resolve a drum name or MIDI number to a valid GM percussion note."""
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        if key in GM_DRUM_NOTES:
            return GM_DRUM_NOTES[key]
        try:
            value = float(value)
        except ValueError:
            return 36
    if value is None:
        return 36
    return max(0, min(127, int(round(float(value)))))
