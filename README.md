# JE AI Audio Studio

AI-assisted audio editing and music production toolkit by **JE Jamal Odasi**.

> This is a brand-new project and is intentionally separate from JE-DAW.

## Current milestone

The project now has a crash-resistant Gradio workspace plus lightweight DSP, neural transcription, neural vocal enhancement, neural pitch analysis/correction and vocal-conditioned music foundations:

- Upload WAV / MP3 / FLAC / OGG / M4A when the runtime supports decoding
- Audio inspection: duration, sample rate, channels, peak and RMS
- Safe mono/stereo handling
- Normalize audio
- Trim audio by start/end time
- Basic Vocal Fix DSP with conservative noise reduction, de-essing, compression and high-pass filtering
- Advanced Vocal Fix controls for noise reduction, de-reverb, pitch correction, timing correction, breath reduction and click/pop cleanup
- Optional **Neural Vocal Enhance** backend using DeepFilterNet
- **Neural Pitch + Timing** backend using pretrained CREPE pitch tracking through torchcrepe
- Vocal → Melody / MIDI extraction
- Optional neural AI MIDI transcription with Spotify Basic Pitch
- **Vocal-conditioned music generation foundation** that estimates key, tempo and activity, then conditions rhythm/chords/bass/drums
- Vocal → Music Parts: rhythm, chords, bass and drums
- Full multi-track MIDI arrangement rendering
- General MIDI percussion-name mapping and Channel 10 drum rendering
- Optional stem separation backend
- Lightweight multi-track audio mixing
- Conservative master-bus compression, saturation and peak limiting
- WAV and MIDI export
- Clear errors instead of crashing the Gradio app

## Optional AI layer

`requirements.txt` stays lightweight for normal development. The optional `requirements-ai.txt` layer adds the neural backends used by Colab/server workflows.

### Neural Vocal Enhance

The Neural Vocal Enhance tab uses the DeepFilterNet Python API for speech/vocal denoising. The wrapper loads a pretrained model lazily, caches it by configuration, processes audio at the model's sample rate, then restores the original sample rate.

### Neural Pitch + Timing

The Neural Pitch + Timing tab uses pretrained CREPE pitch tracking through `torchcrepe` to obtain frame-wise F0 and periodicity. Detected pitch is converted toward the nearest semitone with conservative blockwise correction, while timing can optionally snap detected note onsets toward a BPM grid. This is an intermediate production foundation, not yet a commercial elastic-audio or frame-accurate Auto-Tune clone.

### AI MIDI

The AI MIDI tab uses Spotify Basic Pitch as an optional neural audio-to-MIDI backend. It exposes MIDI tempo, frequency range, minimum note length, onset threshold and frame threshold controls, and returns MIDI note events with pitch-bend information.

### Vocal-conditioned music

`ai/conditioned_music.py` provides a deterministic conditioning layer: it estimates key/scale from chroma, estimates tempo from beat tracking, measures activity from onset strength, and converts those conditions into synchronized rhythm/chord/bass/drum generators. The intent is to provide a stable interface for a later learned arrangement model rather than pretending this stage is already an end-to-end generative model.

Install the optional layer with:

```bash
pip install -r requirements-ai.txt
```

## Planned AI modules

1. Learned/frame-accurate vocal pitch correction and timing alignment
2. Neural vocal restoration beyond denoising
3. Learned Vocal → melody / rhythm / bass / drums / chords generation
4. Stem separation refinement
5. AI arrangement generation and full-song generation
6. Advanced mix and mastering
7. WAV / MP3 / MIDI export pipeline
8. Production web UI and API
9. Android client

## Important implementation note

The current lightweight modules are foundations. The neural backends are real model integrations, but this project is not yet equivalent to a commercial neural vocal editor, neural source separator, or AI music generator. Heavy learned models remain modular so stronger models can be added later.

## Run in Google Colab

Open `colab/JE_AI_Audio_Studio.ipynb` in Colab, run the cells in order, and the notebook will clone/reset the `main` branch, install the base + optional AI dependencies, verify imports, and launch the Gradio UI with a temporary share URL.

## Local

```bash
pip install -r requirements.txt
python app.py
```

For neural features:

```bash
pip install -r requirements-ai.txt
python app.py
```

## Project structure

```text
JE-AI-Audio-Studio/
├── app.py
├── requirements.txt
├── requirements-ai.txt
├── README.md
├── .gitignore
├── colab/
│   └── JE_AI_Audio_Studio.ipynb
├── ai/
│   ├── basic_pitch_transcriber.py
│   ├── conditioned_music.py
│   ├── neural_vocal_enhancement.py
│   ├── pitch_timing_ai.py
│   ├── vocal_to_melody.py
│   └── vocal_to_music.py
├── vocal/
├── separation/
├── music/
├── mixing/
├── utils/
└── ui/
```
