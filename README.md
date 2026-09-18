# JE AI Audio Studio

AI-assisted audio editing and music production toolkit by **JE Jamal Odasi**.

> This is a brand-new project and is intentionally separate from JE-DAW.

## Current milestone

The project now has a crash-resistant Gradio workspace plus lightweight DSP, neural analysis, neural transcription, conditioned arrangement and optional neural audio-generation foundations:

- Upload WAV / MP3 / FLAC / OGG / M4A when the runtime supports decoding
- Audio inspection: duration, sample rate, channels, peak and RMS
- Safe mono/stereo handling
- Normalize audio
- Trim audio by start/end time
- Basic Vocal Fix DSP with conservative noise reduction, de-essing, compression and high-pass filtering
- Advanced Vocal Fix controls for noise reduction, de-reverb, pitch correction, timing correction, breath reduction and click/pop cleanup
- Optional **Neural Vocal Enhance** backend using DeepFilterNet
- Neural **Pitch + Timing** analysis/correction using pretrained CREPE through torchcrepe
- Vocal → Melody / MIDI extraction
- Optional neural AI MIDI transcription with Spotify Basic Pitch
- **AI Conditioned Arrangement**: vocal-conditioned tempo/key/activity → melody + rhythm + chords + bass + drums
- **AI Music Generator**: optional MusicGen Melody text + audio-conditioned short music synthesis
- **AI Song Builder**: Intro / Verse / Chorus / Bridge / Outro section generation with continuity controls, crossfade and section bundle export
- **Full AI Song Pipeline**: vocal cleanup → AI backing → vocal/backing mix → conservative master → final WAV + project ZIP
- **AI Song Builder**: section-by-section Intro → Verse → Chorus → Bridge → Outro sketch generation with crossfades, continuity mode, seeds and a remixable ZIP bundle
- Vocal → Music Parts: rhythm, chords, bass and drums
- Full multi-track MIDI arrangement rendering
- General MIDI percussion-name mapping and Channel 10 drum rendering
- Optional stem separation backend
- **FastAPI service** with health, audio analysis, async song-generation jobs and artifact download endpoints for future Web/Android clients
- Lightweight multi-track audio mixing
- Conservative master-bus compression, saturation and peak limiting
- WAV and MIDI export
- Clear errors instead of crashing the Gradio app

## Optional AI layer

`requirements.txt` stays lightweight for normal development. The optional `requirements-ai.txt` layer adds the neural backends used by Colab/server workflows.

### Neural Vocal Enhance

The Neural Vocal Enhance tab uses the DeepFilterNet Python API for speech/vocal denoising. The wrapper loads the model lazily, caches it by configuration, processes audio at the model sample rate, then resamples the result back to the original sample rate.

### Neural Pitch + Timing

The pitch/timing backend uses pretrained CREPE through `torchcrepe` for frame-wise F0 and periodicity. The current correction stage is deliberately conservative and blockwise; a future learned time-warp/pitch model can replace it without changing the UI contract.

### AI MIDI

The AI MIDI tab uses Spotify Basic Pitch as an optional neural audio-to-MIDI backend. It exposes MIDI tempo, frequency range, minimum note length, onset threshold and frame threshold controls, and returns MIDI note events with pitch-bend information.

### AI Conditioned Arrangement

The arrangement pipeline analyzes source tempo, chroma/key and onset activity, then feeds those conditions into synchronized accompaniment generators. Basic Pitch is used when available for melody extraction, with pYIN fallback. This remains a modular conditioning pipeline rather than an end-to-end learned song model.

### AI Music Generator

The optional MusicGen Melody backend uses the Hugging Face Transformers implementation of `facebook/musicgen-melody` to generate short music from a text description plus an audio/melody reference. The current UI exposes generation duration, guidance scale, temperature, top-k, top-p, seed and device controls.

### AI Song Builder

The Song Builder turns one vocal/melody reference into a structured **song sketch** rather than one undifferentiated audio block. Each selected section is generated with its own musical instruction:

- **Intro** — sparse opening and tonal setup
- **Verse** — restrained support with space for singing
- **Chorus** — fuller lift and stronger rhythmic energy
- **Bridge** — contrast and transition toward the ending
- **Outro** — resolution and gradual release

The sections are crossfaded into a single WAV timeline. `vocal-anchor` keeps the original reference as the conditioning anchor for every section; `chain` feeds the previous generated section into the next section for a more continuous generative chain. The builder also writes a ZIP bundle containing the final sketch, every generated section and a JSON manifest.

The current builder is capped at **90 seconds total** so it behaves as a concept/sketch tool on Colab GPUs rather than pretending to be a full-length production renderer. It does not claim DAW-grade beat-locked continuation or perfect stem continuity.

**Licensing:** Meta's MusicGen model card states that the model weights are released under **CC-BY-NC 4.0**. Treat the bundled MusicGen backend as research/non-commercial unless you substitute a separately licensed model behind the same interface.

Install the optional layer with:

```bash
pip install -r requirements-ai.txt
```

## Planned AI modules

1. Learned frame-accurate pitch correction and time alignment
2. Neural vocal restoration beyond denoising
3. Learned Vocal → melody / rhythm / bass / drums / chords generation
4. Learned style/genre-conditioned arrangement generation
5. Commercially licensable neural audio rendering / full-song synthesis
6. Stem separation refinement
7. Advanced mix and mastering
8. WAV / MP3 / MIDI export pipeline
9. Production web UI and API
10. Android client

## Important implementation note

The current melody, accompaniment, vocal-fix and mix/master modules are lightweight foundations. The neural backends are real optional model integrations, but this project is not yet equivalent to a commercial neural vocal editor, neural source separator, or end-to-end AI music generator. Heavy learned models remain behind modular interfaces so stronger or differently licensed models can be added later.

## Run in Google Colab

Open `colab/JE_AI_Audio_Studio.ipynb` in Colab, run the cells in order, and the notebook will clone/reset the `main` branch, install the base + optional AI dependencies, verify imports, and launch the Gradio UI with a temporary share URL.

For MusicGen generation and Song Builder generation, use a GPU runtime and start with short section durations. The upstream MusicGen documentation describes Melody as text + audio conditioned generation and recommends sampling for practical generation quality.

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

For the optional HTTP API:

```bash
pip install -r requirements-api.txt
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

API endpoints include `GET /health`, `POST /api/analyze`, `POST /api/jobs/song`, `GET /api/jobs/{job_id}`, and artifact downloads under `/api/jobs/{job_id}/download/{artifact}`. The song endpoint returns `202 Accepted` and processes the heavy generation job in a worker thread.

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
│   ├── arrangement_generator.py
│   ├── basic_pitch_transcriber.py
│   ├── conditioned_music.py
│   ├── musicgen_melody.py
│   ├── song_builder.py
│   ├── full_song_pipeline.py
│   ├── neural_vocal_enhancement.py
│   ├── pitch_timing_ai.py
│   ├── song_builder.py
│   ├── vocal_to_melody.py
│   └── vocal_to_music.py
├── api/
│   ├── __init__.py
│   └── server.py
├── vocal/
├── separation/
├── music/
├── mixing/
├── utils/
└── ui/
```
