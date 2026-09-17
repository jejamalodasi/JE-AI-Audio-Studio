# AI backends

The project keeps heavyweight neural runtimes optional so the base CPU/Colab environment stays easier to install.

## Neural Vocal Enhance

`neural_vocal_enhancement.py` wraps DeepFilterNet as an optional neural speech/vocal-enhancement backend. It loads the model lazily, supports automatic CPU/CUDA selection, caches initialized models, runs at the model sample rate, and restores the original input sample rate on export.

Install the optional AI layer:

```bash
pip install -r requirements-ai.txt
```

Use this backend for noisy vocal or speech recordings. It is a denoiser/enhancer, not a source separator for full music mixes. Longer files can require substantial RAM/VRAM.

## Basic Pitch

`basic_pitch_transcriber.py` provides an optional Spotify Basic Pitch backend for neural audio-to-MIDI transcription. Basic Pitch supports polyphonic transcription and pitch-bend-aware MIDI output.

Install when needed:

```bash
pip install basic-pitch
```

Use it for instrument recordings or more complex audio where the current `librosa.pyin` melody extractor is too limited. Basic Pitch works best when the input focuses on one instrument at a time.

The application keeps these backends optional because their runtime dependencies can be heavy and can vary by operating system and Python runtime.
