# AI backends

The project keeps heavyweight neural runtimes optional so the base Colab/CPU environment remains easier to install.

## Basic Pitch

`basic_pitch_transcriber.py` provides an optional Spotify Basic Pitch backend for neural audio-to-MIDI transcription. Basic Pitch is a lightweight automatic music transcription model with polyphonic support and pitch-bend-aware MIDI output.

Install when needed:

```bash
pip install basic-pitch
```

Use it for instrument recordings or more complex audio where the current `librosa.pyin` melody extractor is too limited. Basic Pitch works best when the input focuses on one instrument at a time.

The application keeps this backend optional because its runtime dependencies vary by operating system and Python version.
