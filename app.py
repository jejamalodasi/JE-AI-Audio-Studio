from __future__ import annotations

import tempfile
import traceback
from typing import Optional

import gradio as gr

from ai.vocal_to_melody import MelodyConfig, vocal_to_melody
from ai.vocal_to_music import VocalMusicConfig, generate_from_vocal
from mixing.loudness import audio_stats
from mixing.mastering import MasteringConfig, master_audio
from mixing.mixer import mix_audio_files
from music.bass_generator import BassConfig, generate_bass
from music.chord_generator import ChordConfig, generate_chords
from music.drum_generator import DrumConfig, generate_drums
from music.midi_renderer import render_arrangement_midi
from music.rhythm_generator import RhythmConfig, generate_rhythm
from separation.engine import SeparationConfig, separate_stems
from utils.audio_utils import load_audio, normalize, save_wav, trim_audio
from vocal.analyzer import analyze_vocal
from vocal.vocal_fix import VocalFixConfig, vocal_fix
from vocal.vocal_fix_advanced import AdvancedVocalFixConfig, advanced_vocal_fix


APP_TITLE = "JE AI Audio Studio"

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALES = ["major", "minor"]


def _fmt_db(value: float) -> str:
    return f"{value:.2f} dBFS"


def inspect_audio(path: Optional[str]):
    if not path:
        return "No audio loaded."
    try:
        y, sr = load_audio(path)
        report = analyze_vocal(y, sr)
        return (
            f"**Duration:** {report['duration_seconds']:.2f}s\n\n"
            f"**Sample rate:** {report['sample_rate']:,} Hz\n\n"
            f"**Channels:** {report['channels']}\n\n"
            f"**Peak:** {_fmt_db(report['peak_dbfs'])}\n\n"
            f"**RMS:** {_fmt_db(report['rms_dbfs'])}\n\n"
            f"**Crest factor:** {_fmt_db(report['crest_factor_db'])}\n\n"
            f"**Estimated events:** {report['estimated_events']}"
        )
    except Exception as exc:
        return f"❌ {type(exc).__name__}: {exc}"


def process_audio(path: Optional[str], operation: str, start: float, end: float, nr: float):
    if not path:
        return None, "Please upload an audio file first."
    try:
        y, sr = load_audio(path)
        if operation == "Normalize":
            result = normalize(y)
        elif operation == "Trim":
            result = trim_audio(y, sr, start, end if end > 0 else None)
        elif operation == "Vocal Fix":
            result = vocal_fix(y, sr, VocalFixConfig(noise_reduction=nr))
        else:
            result = y
        out = save_wav(result, sr)
        return out, f"✅ Done — {operation}.\n\nSample rate: {sr:,} Hz\nDuration: {result.shape[0] / sr:.2f}s"
    except Exception as exc:
        traceback.print_exc()
        return None, f"❌ {type(exc).__name__}: {exc}"


def process_advanced_vocal(
    path: Optional[str],
    noise_reduction: float,
    dereverb_strength: float,
    pitch_strength: float,
    timing_strength: float,
    breath_reduction: float,
    click_cleanup: bool,
):
    if not path:
        return None, "Please upload a vocal file first."
    try:
        y, sr = load_audio(path)
        cfg = AdvancedVocalFixConfig(
            noise_reduction=float(noise_reduction),
            dereverb=float(dereverb_strength),
            pitch_correction=float(pitch_strength),
            timing_correction=float(timing_strength),
            breath_reduction=float(breath_reduction),
            click_cleanup=bool(click_cleanup),
        )
        result = advanced_vocal_fix(y, sr, cfg)
        out = save_wav(result, sr)
        stats = audio_stats(result)
        return out, (
            "### 🧪 Advanced Vocal Fix complete\n"
            f"Sample rate: **{sr:,} Hz**\n\n"
            f"Peak: **{stats['peak_dbfs']:.2f} dBFS**\n\n"
            f"RMS: **{stats['rms_dbfs']:.2f} dBFS**\n\n"
            "Applied using the current lightweight DSP foundation."
        )
    except Exception as exc:
        traceback.print_exc()
        return None, f"❌ Advanced Vocal Fix failed: {type(exc).__name__}: {exc}"


def extract_melody(path: Optional[str], bpm: float, fmin: float, fmax: float):
    if not path:
        return None, "Please upload a vocal file first."
    try:
        y, sr = load_audio(path)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            midi_path = tmp.name
        midi_path, notes = vocal_to_melody(
            y,
            sr,
            output_path=midi_path,
            bpm=bpm,
            config=MelodyConfig(fmin=fmin, fmax=fmax),
        )
        preview = notes[:20]
        lines = [f"### 🎼 Melody extracted — {len(notes)} notes", f"BPM: **{bpm:.0f}**"]
        if preview:
            lines.append("\nFirst notes:")
            lines.extend(
                f"- MIDI **{n['note']}** · {n['start']:.2f}s · {n['duration']:.2f}s"
                for n in preview
            )
        return midi_path, "\n".join(lines)
    except Exception as exc:
        traceback.print_exc()
        return None, f"❌ Melody extraction failed: {type(exc).__name__}: {exc}"


def generate_full_arrangement(
    path: Optional[str],
    bpm: float,
    key: str,
    scale: str,
    bars: int,
    fmin: float,
    fmax: float,
    seed: int,
):
    if not path:
        return None, None, "Please upload a vocal file first."
    try:
        cfg = VocalMusicConfig(
            bpm=float(bpm),
            key=str(key),
            scale=str(scale),
            bars=int(bars),
            fmin=float(fmin),
            fmax=float(fmax),
            seed=int(seed),
        )
        result = generate_from_vocal(path, config=cfg)
        summary = [
            "### 🎼 Full MIDI arrangement generated",
            f"BPM: **{cfg.bpm:.0f}**",
            f"Key: **{cfg.key} {cfg.scale}**",
            f"Bars: **{cfg.bars}**",
            f"Melody notes: **{len(result['melody'])}**",
            f"Chords: **{len(result['chords'])}**",
            f"Bass events: **{len(result['bass'])}**",
            f"Drum events: **{len(result['drums'])}**",
            f"Rhythm events: **{len(result['rhythm']['events'])}**",
        ]
        return result["arrangement_midi_path"], result["midi_path"], "\n\n".join(summary)
    except Exception as exc:
        traceback.print_exc()
        return None, None, f"❌ Full arrangement failed: {type(exc).__name__}: {exc}"


def generate_music_parts(
    path: Optional[str],
    bpm: float,
    key: str,
    scale: str,
    bars: int,
    density: float,
    swing: float,
    seed: int,
):
    """Generate separate MIDI files for rhythm, chords, bass, and drums."""
    if not path:
        return None, None, None, None, "Please upload a vocal/audio file first."
    try:
        load_audio(path, mono=True)
        bpm = float(bpm)
        bars = max(1, int(bars))
        seed = int(seed)

        rhythm = generate_rhythm(
            RhythmConfig(
                bpm=bpm,
                bars=bars,
                density=float(density),
                swing=float(swing),
                seed=seed,
            )
        )
        chords = generate_chords(ChordConfig(bpm=bpm, bars=bars, key=str(key), scale=str(scale)))
        bass = generate_bass(BassConfig(bpm=bpm, bars=bars, key=str(key), scale=str(scale)))
        drums = generate_drums(DrumConfig(bpm=bpm, bars=bars, density=float(density), seed=seed))

        def temp_midi(label: str) -> str:
            safe = label.lower().replace(" ", "_")
            handle = tempfile.NamedTemporaryFile(suffix=f"_{safe}.mid", delete=False)
            handle.close()
            return handle.name

        rhythm_path = render_arrangement_midi(None, rhythm, None, None, None, temp_midi("rhythm"), bpm=bpm)
        chord_path = render_arrangement_midi(None, None, chords, None, None, temp_midi("chords"), bpm=bpm)
        bass_path = render_arrangement_midi(None, None, None, bass, None, temp_midi("bass"), bpm=bpm)
        drum_path = render_arrangement_midi(None, None, None, None, drums, temp_midi("drums"), bpm=bpm)

        status = (
            "### 🎛️ Music parts generated\n"
            f"BPM: **{bpm:.0f}** · Key: **{key} {scale}** · Bars: **{bars}**\n\n"
            f"Rhythm events: **{len(rhythm['events'])}**\n\n"
            f"Chords: **{len(chords)}**\n\n"
            f"Bass events: **{len(bass)}**\n\n"
            f"Drum events: **{len(drums)}**\n\n"
            "These are synchronized MIDI building blocks for the next arrangement/model stages."
        )
        return rhythm_path, chord_path, bass_path, drum_path, status
    except Exception as exc:
        traceback.print_exc()
        return None, None, None, None, f"❌ Music part generation failed: {type(exc).__name__}: {exc}"


def mix_and_master(paths, master_gain_db: float, compression_ratio: float, saturation: float, target_peak: float):
    """Mix uploaded stems and run the lightweight master bus."""
    if not paths:
        return None, "Please upload at least one audio file."
    try:
        file_paths = list(paths) if isinstance(paths, (list, tuple)) else [paths]
        file_paths = [str(p) for p in file_paths if p]
        if not file_paths:
            return None, "Please upload at least one audio file."

        mixed_path = mix_audio_files(file_paths)
        mixed, sr = load_audio(mixed_path)
        gain = 10.0 ** (float(master_gain_db) / 20.0)
        mixed = mixed * gain
        cfg = MasteringConfig(
            target_peak=float(target_peak),
            compressor_ratio=float(compression_ratio),
            saturation=float(saturation),
        )
        mastered = master_audio(mixed, cfg)
        out = save_wav(mastered, sr)
        stats = audio_stats(mastered)
        return out, (
            "### 🎚️ Mix & Master complete\n"
            f"Tracks mixed: **{len(file_paths)}**\n\n"
            f"Sample rate: **{sr:,} Hz**\n\n"
            f"Peak: **{stats['peak_dbfs']:.2f} dBFS**\n\n"
            f"RMS: **{stats['rms_dbfs']:.2f} dBFS**\n\n"
            f"Crest: **{stats['crest_db']:.2f} dB**"
        )
    except Exception as exc:
        traceback.print_exc()
        return None, f"❌ Mix/Master failed: {type(exc).__name__}: {exc}"


def separate_audio(path: Optional[str], model: str, shifts: int, overlap: float):
    if not path:
        return [], "Please upload an audio file first."
    try:
        outputs = separate_stems(
            path,
            config=SeparationConfig(model=model, shifts=int(shifts), overlap=float(overlap)),
        )
        files = list(outputs.values())
        names = ", ".join(outputs.keys())
        return files, f"### 🧩 Separation complete\nSources: **{names}**\n\nThe files above are the exported WAV stems."
    except Exception as exc:
        traceback.print_exc()
        return [], f"❌ Stem separation failed: {type(exc).__name__}: {exc}"


def build_app():
    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎚️ JE AI Audio Studio\n"
            "### AI-assisted audio editing & music production\n\n"
            "Audio engine + Vocal Fix DSP + Advanced Vocal Fix + Vocal→MIDI + Full Arrangement + Mix/Master + optional stem separation."
        )

        with gr.Tabs():
            with gr.Tab("🎚️ Audio / Vocal Fix"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_in = gr.Audio(label="Input Audio", type="filepath", sources=["upload", "microphone"])
                        inspect_btn = gr.Button("🔎 Analyze Audio", variant="secondary")
                        info = gr.Markdown("Upload a file and press **Analyze Audio**.")
                    with gr.Column(scale=1):
                        operation = gr.Radio(["Vocal Fix", "Normalize", "Trim", "Pass-through"], value="Vocal Fix", label="Operation")
                        nr = gr.Slider(0.0, 2.0, value=0.65, step=0.05, label="Noise Reduction Strength")
                        with gr.Row():
                            start = gr.Number(value=0, minimum=0, label="Start (seconds)")
                            end = gr.Number(value=0, minimum=0, label="End (seconds, 0 = file end)")
                        process_btn = gr.Button("⚡ Process", variant="primary")
                        output = gr.File(label="Processed WAV")
                        status = gr.Markdown()
                inspect_btn.click(inspect_audio, inputs=audio_in, outputs=info)
                process_btn.click(process_audio, inputs=[audio_in, operation, start, end, nr], outputs=[output, status])

            with gr.Tab("🧪 Advanced Vocal Fix"):
                advanced_in = gr.Audio(label="Vocal Input", type="filepath", sources=["upload", "microphone"])
                with gr.Row():
                    advanced_nr = gr.Slider(0.0, 2.0, value=0.65, step=0.05, label="Noise Reduction")
                    advanced_dereverb = gr.Slider(0.0, 1.0, value=0.30, step=0.05, label="De-Reverb")
                    advanced_pitch = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Pitch Correction")
                with gr.Row():
                    advanced_timing = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Timing Correction")
                    advanced_breath = gr.Slider(0.0, 1.0, value=0.20, step=0.05, label="Breath Reduction")
                    advanced_click = gr.Checkbox(value=True, label="Click / Pop Cleanup")
                advanced_btn = gr.Button("🧪 Run Advanced Vocal Fix", variant="primary")
                advanced_out = gr.File(label="Advanced Fixed WAV")
                advanced_status = gr.Markdown(
                    "Use small pitch/timing values first. The current pitch/timing modules are conservative DSP foundations, not frame-wise commercial Auto-Tune."
                )
                advanced_btn.click(
                    process_advanced_vocal,
                    inputs=[advanced_in, advanced_nr, advanced_dereverb, advanced_pitch, advanced_timing, advanced_breath, advanced_click],
                    outputs=[advanced_out, advanced_status],
                )

            with gr.Tab("🎼 Vocal → Melody / MIDI"):
                melody_in = gr.Audio(label="Vocal Input", type="filepath", sources=["upload", "microphone"])
                with gr.Row():
                    bpm = gr.Slider(40, 240, value=120, step=1, label="BPM")
                    fmin = gr.Number(value=65.41, label="Minimum pitch (Hz)")
                    fmax = gr.Number(value=1046.50, label="Maximum pitch (Hz)")
                melody_btn = gr.Button("🎼 Extract Melody → MIDI", variant="primary")
                melody_out = gr.File(label="MIDI File")
                melody_status = gr.Markdown("Upload a mostly-monophonic vocal, then extract its melody.")
                melody_btn.click(extract_melody, inputs=[melody_in, bpm, fmin, fmax], outputs=[melody_out, melody_status])

            with gr.Tab("🎛️ Vocal → Music Parts"):
                parts_in = gr.Audio(label="Vocal / Audio Input", type="filepath", sources=["upload", "microphone"])
                with gr.Row():
                    parts_bpm = gr.Slider(40, 240, value=120, step=1, label="BPM")
                    parts_key = gr.Dropdown(KEYS, value="C", label="Key")
                    parts_scale = gr.Dropdown(SCALES, value="major", label="Scale")
                    parts_bars = gr.Slider(1, 64, value=4, step=1, label="Bars")
                with gr.Row():
                    parts_density = gr.Slider(0.0, 1.0, value=0.55, step=0.05, label="Rhythm/Drum Density")
                    parts_swing = gr.Slider(-0.5, 0.5, value=0.0, step=0.05, label="Swing")
                    parts_seed = gr.Number(value=42, precision=0, label="Seed")
                parts_btn = gr.Button("🎛️ Generate Music Parts", variant="primary")
                with gr.Row():
                    rhythm_out = gr.File(label="Rhythm MIDI")
                    chord_out = gr.File(label="Chords MIDI")
                with gr.Row():
                    bass_out = gr.File(label="Bass MIDI")
                    drum_out = gr.File(label="Drums MIDI")
                parts_status = gr.Markdown("Generates synchronized rhythm, chord, bass and GM-drum MIDI building blocks.")
                parts_btn.click(
                    generate_music_parts,
                    inputs=[parts_in, parts_bpm, parts_key, parts_scale, parts_bars, parts_density, parts_swing, parts_seed],
                    outputs=[rhythm_out, chord_out, bass_out, drum_out, parts_status],
                )

            with gr.Tab("🎼 Vocal → Full Arrangement"):
                arrange_in = gr.Audio(label="Vocal Input", type="filepath", sources=["upload", "microphone"])
                with gr.Row():
                    arrange_bpm = gr.Slider(40, 240, value=120, step=1, label="BPM")
                    arrange_key = gr.Dropdown(KEYS, value="C", label="Key")
                    arrange_scale = gr.Dropdown(SCALES, value="major", label="Scale")
                with gr.Row():
                    arrange_bars = gr.Slider(1, 64, value=4, step=1, label="Bars")
                    arrange_fmin = gr.Number(value=65.41, label="Minimum pitch (Hz)")
                    arrange_fmax = gr.Number(value=1046.50, label="Maximum pitch (Hz)")
                    arrange_seed = gr.Number(value=42, precision=0, label="Seed")
                arrange_btn = gr.Button("🎼 Generate Full Arrangement", variant="primary")
                arrangement_out = gr.File(label="Full Arrangement MIDI")
                melody_arrangement_out = gr.File(label="Melody-only MIDI")
                arrangement_status = gr.Markdown(
                    "Generates melody + rhythm + chords + bass + drums as one synchronized MIDI arrangement."
                )
                arrange_btn.click(
                    generate_full_arrangement,
                    inputs=[arrange_in, arrange_bpm, arrange_key, arrange_scale, arrange_bars, arrange_fmin, arrange_fmax, arrange_seed],
                    outputs=[arrangement_out, melody_arrangement_out, arrangement_status],
                )

            with gr.Tab("🎚️ Mix & Master"):
                mix_in = gr.Files(label="Upload Stems / Tracks", file_count="multiple", type="filepath")
                with gr.Row():
                    master_gain = gr.Slider(-12, 12, value=0, step=0.5, label="Master Gain (dB)")
                    compression = gr.Slider(1.0, 6.0, value=2.0, step=0.1, label="Compressor Ratio")
                    saturation = gr.Slider(0.0, 0.5, value=0.08, step=0.01, label="Saturation")
                    target_peak = gr.Slider(0.8, 0.99, value=0.95, step=0.01, label="Target Peak")
                mix_btn = gr.Button("🎚️ Mix + Master", variant="primary")
                mix_out = gr.File(label="Mastered WAV")
                mix_status = gr.Markdown(
                    "Upload vocals/instruments/stems. The current engine performs a lightweight stereo sum followed by conservative bus compression, saturation and peak limiting."
                )
                mix_btn.click(
                    mix_and_master,
                    inputs=[mix_in, master_gain, compression, saturation, target_peak],
                    outputs=[mix_out, mix_status],
                )

            with gr.Tab("🧩 Stem Separation"):
                stem_in = gr.Audio(label="Song / Mix Input", type="filepath", sources=["upload"])
                with gr.Row():
                    model = gr.Dropdown(["htdemucs", "htdemucs_ft"], value="htdemucs", label="Model")
                    shifts = gr.Slider(0, 2, value=1, step=1, label="Quality shifts")
                    overlap = gr.Slider(0.1, 0.75, value=0.25, step=0.05, label="Overlap")
                separate_btn = gr.Button("🧩 Separate Stems", variant="primary")
                stem_outputs = gr.Files(label="Separated WAV Stems")
                stem_status = gr.Markdown("Stem separation is an optional heavy ML feature; GPU/Colab is recommended.")
                separate_btn.click(separate_audio, inputs=[stem_in, model, shifts, overlap], outputs=[stem_outputs, stem_status])

        gr.Markdown(
            "---\n### 🧠 Engine roadmap\n"
            "✅ Audio analysis · ✅ Vocal Fix DSP · ✅ Advanced Vocal Fix · ✅ Vocal→Melody/MIDI · ✅ Music Parts · ✅ Stem separation backend · ✅ Full MIDI Arrangement · ✅ Mix/Master foundation\n\n"
            "Next: **AI-conditioned vocal/music models → production web UI/API → Android client.**"
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
