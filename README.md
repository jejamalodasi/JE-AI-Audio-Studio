# JE AI Audio Studio

AI-assisted audio editing and music production toolkit by **JE Jamal Odasi**.

> This is a brand-new project and is intentionally separate from JE-DAW.

## Current milestone

Phase 1 establishes a crash-resistant Gradio audio workspace:

- Upload WAV / MP3 / FLAC / OGG / M4A when the runtime supports decoding
- Audio inspection: duration, sample rate, channels, peak and RMS
- Safe mono/stereo handling
- Normalize audio
- Trim audio by start/end time
- Export WAV
- Clear errors instead of crashing the Gradio app
- Temporary-file cleanup

## Planned AI modules

1. Vocal analysis and one-click Vocal Fix
2. Noise reduction / de-reverb / de-esser / breath-click-pop cleanup
3. Pitch and timing correction
4. Vocal → MIDI / melody / rhythm / bass / drums / chords
5. Stem separation
6. Arrangement generation
7. Mix and mastering
8. WAV / MP3 / MIDI export
9. Android client and production API

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
