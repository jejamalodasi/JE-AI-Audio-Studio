# AI backends

The project keeps heavyweight neural runtimes optional so the base CPU environment remains easier to install.

## Basic Pitch

`basic_pitch_transcriber.py` provides an optional Spotify Basic Pitch backend for neural audio-to-MIDI transcription.

## DeepFilterNet

`neural_vocal_enhancement.py` provides optional DeepFilterNet-based vocal/speech enhancement. The model is loaded lazily and cached by configuration.

## Neural Pitch + Timing

`pitch_timing_ai.py` uses pretrained CREPE through `torchcrepe` for frame-wise F0/periodicity analysis, followed by a conservative blockwise pitch correction and BPM-grid timing warp.

## Conditioned music

`conditioned_music.py` extracts tempo, chroma/key and onset activity from the source audio and converts those observations into synchronized rhythm, chord, bass and drum controls.

## AI Conditioned Arrangement

`arrangement_generator.py` combines the conditioning layer with melody extraction. In `auto` mode it tries Basic Pitch first and falls back to the existing pYIN melody extractor when the optional Basic Pitch backend is not available. The final arrangement is rendered as multi-track MIDI.

## MusicGen Melody

`musicgen_melody.py` provides an optional Transformers-based `facebook/musicgen-melody` audio-generation backend. It accepts an audio/melody reference plus a text description and generates a short music waveform. The model supports audio-conditioned and text-conditioned generation through the Hugging Face Transformers API.

The bundled MusicGen weights are released under **CC-BY-NC 4.0**, so this backend should be treated as research/non-commercial unless a separately licensed model is substituted behind the same interface.

Install the optional AI stack:

```bash
pip install -r requirements-ai.txt
```
