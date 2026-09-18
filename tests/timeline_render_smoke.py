from pathlib import Path
import tempfile

import mido
import numpy as np
import soundfile as sf

from music.midi_audio_renderer import render_midi_to_audio


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        midi_path = root / "smoke.mid"
        wav_path = root / "smoke.wav"

        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
        track.append(mido.Message("note_on", note=60, velocity=100, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, time=480))
        track.append(mido.Message("note_on", note=64, velocity=100, time=0))
        track.append(mido.Message("note_off", note=64, velocity=0, time=480))
        mid.save(midi_path)

        render_midi_to_audio(str(midi_path), str(wav_path), bpm=120, sample_rate=22050)

        audio, sr = sf.read(wav_path, always_2d=True)
        assert sr == 22050
        assert audio.shape[1] == 2
        assert audio.shape[0] > 20000
        assert float(np.max(np.abs(audio))) > 0.01

        print("PASS: MIDI -> WAV render")
        print("sample_rate:", sr)
        print("channels:", audio.shape[1])
        print("duration_seconds:", round(audio.shape[0] / sr, 3))
        print("peak:", round(float(np.max(np.abs(audio))), 4))


if __name__ == "__main__":
    main()
