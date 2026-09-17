# JE AI Audio Studio

AI-assisted audio editing and music production toolkit by **JE Jamal Odasi**.

> This is a brand-new project and is intentionally separate from JE-DAW.

## Current milestone

The project now has a crash-resistant Gradio workspace plus lightweight vocal, music-generation and mix/master foundations:

- Upload WAV / MP3 / FLAC / OGG / M4A when the runtime supports decoding
- Audio inspection: duration, sample rate, channels, peak and RMS
- Safe mono/stereo handling
- Normalize audio
- Trim audio by start/end time
- Basic Vocal Fix DSP with conservative noise reduction, de-essing, compression and high-pass filtering
- Advanced Vocal Fix controls for noise reduction, de-reverb, pitch correction, timing correction, breath reduction and click/pop cleanup
- Vocal → Melody / MIDI extraction
- Vocal → Music Parts: rhythm, chords, bass and drums
- Full multi-track MIDI arrangement rendering
- General MIDI percussion-name mapping and Channel 10 drum rendering
- Optional stem separation backend
- Lightweight multi-track audio mixing
- Conservative master-bus compression, saturation and peak limiting
- WAV and MIDI export
- Clear errors instead of crashing the Gradio app

## Planned AI modules

1. Neural/ML vocal cleanup and restoration
2. Frame-wise pitch correction and timing alignment
3. AI-conditioned Vocal → melody / rhythm / bass / drums / chords
4. Stem separation refinement
5. AI arrangement generation and full-song generation
6. Advanced mix and mastering
7. WAV / MP3 / MIDI export pipeline
8. Production web UI and API
9. Android client

## Important implementation note

The current melody, accompaniment, vocal-fix and mix/master modules are lightweight foundations designed for CPU/Colab development. They are not yet equivalent to a commercial neural vocal editor, neural source separator, or AI music generator. Heavy learned models can be added behind the same module interfaces later.

## Run in Google Colab

Open `colab/JE_AI_Audio_Studio.ipynb` in Colab, run the setup cell, then launch the app.

## Local

```bash
pip install -r requirements.txt
python app.py
```

## Project structure

```text
JE-AI-Audio-Studio/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── colab/
├── ai/
├── vocal/
├── separation/
├── music/
├── mixing/
├── utils/
└── ui/
```
