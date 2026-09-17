from __future__ import annotations

import tempfile
import traceback
from typing import Optional

import gradio as gr

from ai.arrangement_generator import AIConditionedArrangementConfig, generate_ai_conditioned_arrangement
from ai.basic_pitch_transcriber import transcribe_with_basic_pitch
from ai.musicgen_melody import MusicGenConfig, generate_musicgen_melody
from ai.neural_vocal_enhancement import enhance_vocal_neural
from ai.pitch_timing_ai import PitchTimingAIConfig, correct_pitch_timing_ai
from ai.song_builder import SECTION_ORDER, SongBuilderConfig, SongSection, build_song_sketch
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
            result = vocal_fix(y, sr, VocalFixConfig(noise_reduction=float(nr)))
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
            f"RMS: **{stats['rms_dbfs']:.2f} dBFS**"
        )
    except Exception as exc:
        traceback.print_exc()
        return None, f"❌ Advanced Vocal Fix failed: {type(exc).__name__}: {exc}"


def enhance_vocal_with_ai(path: Optional[str], model_name: str, device: str, atten_lim_db: float, post_filter: bool):
    if not path:
        return None, "Please upload a vocal/audio file first."
    try:
        result = enhance_vocal_neural(
            path,
            model_name=str(model_name),
            device=str(device),
            post_filter=bool(post_filter),
            atten_lim_db=float(atten_lim_db),
        )
        return result["output_path"], (
            "### 🧠 Neural Vocal Enhance complete\n"
            f"Backend: **DeepFilterNet**\n\n"
            f"Model: **{result['model']}**\n\n"
            f"Device: **{result['device']}**\n\n"
            f"Sample rate restored to: **{result['sample_rate']:,} Hz**\n\n"
            f"Duration: **{result['duration_seconds']:.2f}s**"
        )
    except Exception as exc:
        traceback.print_exc()
        return None, f"❌ Neural Vocal Enhance failed: {type(exc).__name__}: {exc}"


def correct_pitch_timing_with_ai(
    path: Optional[str],
    fmin: float,
    fmax: float,
    model: str,
    device: str,
    periodicity: float,
    pitch_strength: float,
    max_semitones: float,
    block_ms: float,
    timing_strength: float,
    bpm: float,
    max_timing_shift_ms: float,
):
    if not path:
        return None, "Please upload a vocal file first."
    try:
        y, sr = load_audio(path)
        cfg = PitchTimingAIConfig(
            fmin=float(fmin),
            fmax=float(fmax),
            model=str(model),
            periodicity_threshold=float(periodicity),
            correction_strength=float(pitch_strength),
            max_semitones=float(max_semitones),
            block_ms=float(block_ms),
            timing_strength=float(timing_strength),
            bpm=float(bpm),
            max_timing_shift_ms=float(max_timing_shift_ms),
        )
        corrected, report = correct_pitch_timing_ai(y, sr, cfg, device=str(device))
        out = save_wav(corrected, sr)
        return out, (
            "### 🎯 Neural Pitch + Timing complete\n"
            f"Backend: **Torch-CREPE / torchcrepe**\n\n"
            f"Device: **{report['device']}**\n\n"
            f"Voiced frames: **{report['voiced_frames']}**\n\n"
            f"Detected note onsets: **{len(report['note_onsets'])}**\n\n"
            "Correction is deliberately conservative; this is not a commercial frame-accurate Auto-Tune/elastic-audio engine yet."
        )
    except Exception as exc:
        traceback.print_exc()
        return None, f"❌ Neural Pitch/Timing failed: {type(exc).__name__}: {exc}"


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
            bpm=float(bpm),
            config=MelodyConfig(fmin=float(fmin), fmax=float(fmax)),
        )
        lines = [f"### 🎼 Melody extracted — {len(notes)} notes", f"BPM: **{float(bpm):.0f}**"]
        if notes:
            lines.append("\nFirst notes:")
            lines.extend(
                f"- MIDI **{n['note']}** · {n['start']:.2f}s · {n['duration']:.2f}s" for n in notes[:20]
            )
        return midi_path, "\n".join(lines)
    except Exception as exc:
        traceback.print_exc()
        return None, f"❌ Melody extraction failed: {type(exc).__name__}: {exc}"


def transcribe_ai_midi(
    path: Optional[str],
    min_freq: float,
    max_freq: float,
    min_note_ms: float,
    onset: float,
    frame: float,
    bpm: float,
):
    if not path:
        return None, "Please upload an audio file first."
    try:
        result = transcribe_with_basic_pitch(
            path,
            minimum_frequency=float(min_freq) if float(min_freq or 0) > 0 else None,
            maximum_frequency=float(max_freq) if float(max_freq or 0) > 0 else None,
            minimum_note_length_ms=max(1.0, float(min_note_ms)),
            onset_threshold=float(onset),
            frame_threshold=float(frame),
            midi_tempo=float(bpm),
        )
        preview = result["notes"][:12]
        lines = [
            "### 🧠 AI MIDI complete",
            "Backend: **Spotify Basic Pitch**",
            f"Notes detected: **{result['note_count']}**",
            f"MIDI tempo: **{float(bpm):.0f} BPM**",
        ]
        if preview:
            lines.append("\nFirst detected notes:")
            lines.extend(
                f"- MIDI **{n['note']}** · {n['start']:.2f}s → {n['end']:.2f}s · velocity {n['velocity']:.2f}"
                for n in preview
            )
        return result["midi_path"], "\n\n".join(lines)
    except Exception as exc:
        traceback.print_exc()
        return None, f"❌ AI MIDI failed: {type(exc).__name__}: {exc}"


def generate_ai_arrangement(
    path: Optional[str],
    bpm_override: float,
    bars: int,
    key: str,
    scale: str,
    melody_backend: str,
    min_freq: float,
    max_freq: float,
    min_note_ms: float,
    onset: float,
    frame: float,
    density_floor: float,
    density_ceiling: float,
    swing: float,
    seed: int,
):
    if not path:
        return None, None, "Please upload a vocal/audio file first."
    try:
        bpm_value = float(bpm_override)
        config = AIConditionedArrangementConfig(
            bpm=bpm_value if bpm_value > 0 else None,
            bars=max(1, int(bars)),
            seed=int(seed),
            key=None if str(key) == "Auto" else str(key),
            scale=None if str(scale) == "Auto" else str(scale),
            melody_backend=str(melody_backend),
            min_frequency=max(0.0, float(min_freq)),
            max_frequency=max(0.0, float(max_freq)),
            min_note_length_ms=max(1.0, float(min_note_ms)),
            onset_threshold=float(onset),
            frame_threshold=float(frame),
            density_floor=float(density_floor),
            density_ceiling=float(density_ceiling),
            swing=float(swing),
        )
        result = generate_ai_conditioned_arrangement(path, config=config)
        status = (
            "### 🤖 AI Conditioned Arrangement complete\n"
            f"Detected/selected tempo: **{result['bpm']:.1f} BPM**\n\n"
            f"Key: **{result['key']} {result['scale']}** · confidence **{result['key_confidence']:.3f}**\n\n"
            f"Vocal activity: **{result['activity']:.3f}** · generated density **{result['density']:.3f}**\n\n"
            f"Melody backend: **{result['melody_backend']}** · notes **{len(result['melody'])}**\n\n"
            "Tracks: **melody + rhythm + chords + bass + drums**"
        )
        return result["arrangement_path"], result["melody_path"], status
    except Exception as exc:
        traceback.print_exc()
        return None, None, f"❌ AI arrangement failed: {type(exc).__name__}: {exc}"


def generate_musicgen_audio(
    path: Optional[str],
    prompt: str,
    duration: float,
    guidance: float,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    device: str,
):
    if not path:
        return None, "Please upload a vocal/melody reference first."
    try:
        result = generate_musicgen_melody(
            path,
            str(prompt),
            config=MusicGenConfig(
                duration_seconds=float(duration),
                guidance_scale=float(guidance),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                seed=int(seed),
                device=str(device),
            ),
        )
        return result["output_path"], (
            "### 🎵 AI Music Generator complete\n"
            f"Backend: **MusicGen Melody / Transformers**\n\n"
            f"Model: **{result['model']}**\n\n"
            f"Device: **{result['device']}**\n\n"
            f"Output: **{result['duration_seconds']:.2f}s @ {result['sampling_rate']:,} Hz**\n\n"
            "⚠️ The bundled MusicGen weights are CC-BY-NC 4.0. Use a separately licensed model for commercial deployment."
        )
    except Exception as exc:
        traceback.print_exc()
        return None, f"❌ AI Music generation failed: {type(exc).__name__}: {exc}"


def build_ai_song_sketch(
    path: Optional[str],
    prompt: str,
    bpm: float,
    key: str,
    scale: str,
    selected_sections,
    intro_seconds: float,
    verse_seconds: float,
    chorus_seconds: float,
    bridge_seconds: float,
    outro_seconds: float,
    crossfade: float,
    guidance: float,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    continuity: str,
    device: str,
):
    if not path:
        return None, None, "Please upload a vocal/melody reference first."
    try:
        selected = [str(s) for s in (selected_sections or [])]
        if not selected:
            raise ValueError("Select at least one song section.")

        durations = {
            "Intro": float(intro_seconds),
            "Verse": float(verse_seconds),
            "Chorus": float(chorus_seconds),
            "Bridge": float(bridge_seconds),
            "Outro": float(outro_seconds),
        }
        sections = tuple(SongSection(name, durations[name]) for name in SECTION_ORDER if name in selected)
        cfg = SongBuilderConfig(
            base_prompt=str(prompt).strip(),
            bpm=float(bpm) if float(bpm) > 0 else None,
            key=None if str(key) == "Auto" else str(key),
            scale=None if str(scale) == "Auto" else str(scale),
            sections=sections,
            crossfade_seconds=float(crossfade),
            guidance_scale=float(guidance),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            seed=int(seed),
            device=str(device),
            continuity=str(continuity),
            max_total_seconds=90.0,
        )
        result = build_song_sketch(path, config=cfg)
        section_lines = "\n".join(
            f"- **{s['name']}** · {s['actual_duration_seconds']:.2f}s · seed {s['seed']}"
            for s in result["sections"]
        )
        status = (
            "### 🎹 AI Song Builder complete\n"
            f"Sections: **{result['section_count']}**\n\n"
            f"Final sketch: **{result['duration_seconds']:.2f}s @ {result['sampling_rate']:,} Hz**\n\n"
            f"Continuity: **{result['continuity']}**\n\n"
            f"Crossfade: **{float(crossfade):.2f}s**\n\n"
            f"{section_lines}\n\n"
            "⚠️ MusicGen weights are CC-BY-NC 4.0. For commercial deployment, replace the backend with a separately licensed model."
        )
        return result["output_path"], result["bundle_path"], status
    except Exception as exc:
        traceback.print_exc()
        return None, None, f"❌ AI Song Builder failed: {type(exc).__name__}: {exc}"


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
    if not path:
        return None, None, None, None, "Please upload a vocal/audio file first."
    try:
        load_audio(path, mono=True)
        bpm_value, bars_value, seed_value = float(bpm), max(1, int(bars)), int(seed)
        rhythm = generate_rhythm(
            RhythmConfig(bpm=bpm_value, bars=bars_value, density=float(density), swing=float(swing), seed=seed_value)
        )
        chords = generate_chords(ChordConfig(bpm=bpm_value, bars=bars_value, key=str(key), scale=str(scale)))
        bass = generate_bass(BassConfig(bpm=bpm_value, bars=bars_value, key=str(key), scale=str(scale)))
        drums = generate_drums(DrumConfig(bpm=bpm_value, bars=bars_value, density=float(density), seed=seed_value))
        with tempfile.TemporaryDirectory(prefix="je_music_parts_") as temp:
            rhythm_path = render_arrangement_midi([], rhythm, [], [], [], f"{temp}/rhythm.mid", bpm=bpm_value)
            chord_path = render_arrangement_midi([], None, chords, [], [], f"{temp}/chords.mid", bpm=bpm_value)
            bass_path = render_arrangement_midi([], None, [], bass, [], f"{temp}/bass.mid", bpm=bpm_value)
            drum_path = render_arrangement_midi([], None, [], [], drums, f"{temp}/drums.mid", bpm=bpm_value)
            persistent = []
            for src in [rhythm_path, chord_path, bass_path, drum_path]:
                with open(src, "rb") as handle:
                    data = handle.read()
                with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as target:
                    target.write(data)
                    persistent.append(target.name)
        return (*persistent, "### 🎛️ Music parts generated\nSynchronized rhythm, chord, bass and GM-drum MIDI blocks are ready.")
    except Exception as exc:
        traceback.print_exc()
        return None, None, None, None, f"❌ Music part generation failed: {type(exc).__name__}: {exc}"


def mix_and_master(paths, master_gain_db: float, compression_ratio: float, saturation: float, target_peak: float):
    if not paths:
        return None, "Please upload at least one audio file."
    try:
        file_paths = list(paths) if isinstance(paths, (list, tuple)) else [paths]
        file_paths = [str(p) for p in file_paths if p]
        if not file_paths:
            raise ValueError("No valid audio files were supplied.")
        mixed_path = mix_audio_files(file_paths)
        y, sr = load_audio(mixed_path)
        gain = 10.0 ** (float(master_gain_db) / 20.0)
        y = y * gain
        mastered = master_audio(
            y,
            MasteringConfig(
                target_peak=float(target_peak),
                compressor_ratio=float(compression_ratio),
                saturation=float(saturation),
                makeup_db=1.0,
            ),
        )
        out = save_wav(mastered, sr)
        stats = audio_stats(mastered)
        return out, (
            "### 🎚️ Mix + Master complete\n"
            f"Tracks mixed: **{len(file_paths)}**\n\n"
            f"Peak: **{stats['peak_dbfs']:.2f} dBFS**\n\n"
            f"RMS: **{stats['rms_dbfs']:.2f} dBFS**"
        )
    except Exception as exc:
        traceback.print_exc()
        return None, f"❌ Mix/Master failed: {type(exc).__name__}: {exc}"


def separate_audio(path: Optional[str], model: str, shifts: int, overlap: float, device: str):
    if not path:
        return [], "Please upload an audio file first."
    try:
        outputs = separate_stems(
            path,
            config=SeparationConfig(model=str(model), shifts=int(shifts), overlap=float(overlap), device=str(device)),
        )
        return list(outputs.values()), "### 🧩 Separation complete\n" + "\n".join(f"- {k}: `{v}`" for k, v in outputs.items())
    except Exception as exc:
        traceback.print_exc()
        return [], f"❌ Stem separation failed: {type(exc).__name__}: {exc}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            "# 🎚️ JE AI Audio Studio\n"
            "### AI-assisted vocal repair, MIDI extraction, arrangement, music generation and song sketching\n\n"
            "A modular production playground for Colab/GPU first, with optional heavy neural backends."
        )

        with gr.Tabs():
            with gr.Tab("🎚️ Audio / Vocal Fix"):
                with gr.Row():
                    with gr.Column():
                        audio_in = gr.Audio(label="Input Audio", type="filepath", sources=["upload", "microphone"])
                        inspect_btn = gr.Button("🔎 Analyze Audio", variant="secondary")
                        info = gr.Markdown("Upload a file and press **Analyze Audio**.")
                    with gr.Column():
                        operation = gr.Dropdown(["Normalize", "Trim", "Vocal Fix"], value="Vocal Fix", label="Operation")
                        start = gr.Number(value=0.0, minimum=0.0, label="Trim start (s)")
                        end = gr.Number(value=0.0, minimum=0.0, label="Trim end (s, 0 = end)")
                        nr = gr.Slider(0.0, 2.0, value=0.65, step=0.05, label="Noise reduction")
                        process_btn = gr.Button("⚙️ Process Audio", variant="primary")
                        output = gr.File(label="Processed WAV")
                inspect_btn.click(inspect_audio, inputs=audio_in, outputs=info)
                process_btn.click(process_audio, inputs=[audio_in, operation, start, end, nr], outputs=[output, info])

            with gr.Tab("🧪 Advanced Vocal Fix"):
                advanced_in = gr.Audio(label="Vocal Input", type="filepath", sources=["upload", "microphone"])
                with gr.Row():
                    advanced_nr = gr.Slider(0.0, 2.0, value=0.65, step=0.05, label="Noise Reduction")
                    advanced_dereverb = gr.Slider(0.0, 1.0, value=0.30, step=0.05, label="De-Reverb")
                    advanced_pitch = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Pitch Correction")
                    advanced_timing = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Timing Correction")
                with gr.Row():
                    advanced_breath = gr.Slider(0.0, 1.0, value=0.20, step=0.05, label="Breath Reduction")
                    advanced_click = gr.Checkbox(value=True, label="Click Cleanup")
                advanced_btn = gr.Button("🧪 Run Advanced Vocal Fix", variant="primary")
                advanced_out = gr.File(label="Advanced Fixed WAV")
                advanced_status = gr.Markdown()
                advanced_btn.click(process_advanced_vocal, inputs=[advanced_in, advanced_nr, advanced_dereverb, advanced_pitch, advanced_timing, advanced_breath, advanced_click], outputs=[advanced_out, advanced_status])

            with gr.Tab("🧠 Neural Vocal Enhance"):
                neural_in = gr.Audio(label="Vocal / Speech Input", type="filepath", sources=["upload", "microphone"])
                with gr.Row():
                    neural_model = gr.Dropdown(["DeepFilterNet3"], value="DeepFilterNet3", label="Neural Model")
                    neural_device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto", label="Device")
                    neural_atten = gr.Slider(0.0, 30.0, value=12.0, step=1.0, label="Attenuation limit (dB)")
                    neural_post = gr.Checkbox(value=True, label="Post filter")
                neural_btn = gr.Button("🧠 Enhance Vocal with Neural AI", variant="primary")
                neural_out = gr.File(label="Neural Enhanced WAV")
                neural_status = gr.Markdown("Optional DeepFilterNet backend for noisy vocal/speech enhancement. GPU/Colab is recommended for longer files.")
                neural_btn.click(enhance_vocal_with_ai, inputs=[neural_in, neural_model, neural_device, neural_atten, neural_post], outputs=[neural_out, neural_status])

            with gr.Tab("🎯 Neural Pitch + Timing"):
                pt_in = gr.Audio(label="Vocal Input", type="filepath", sources=["upload", "microphone"])
                with gr.Row():
                    pt_fmin = gr.Number(value=65.0, minimum=20, maximum=2000, label="Minimum F0 (Hz)")
                    pt_fmax = gr.Number(value=1100.0, minimum=100, maximum=3000, label="Maximum F0 (Hz)")
                    pt_model = gr.Dropdown(["tiny", "full"], value="full", label="CREPE model")
                    pt_device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto", label="Device")
                with gr.Row():
                    pt_periodicity = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Periodicity threshold")
                    pt_pitch = gr.Slider(0.0, 1.0, value=0.65, step=0.05, label="Pitch correction strength")
                    pt_max_semitones = gr.Slider(0.1, 4.0, value=2.0, step=0.1, label="Max semitone shift")
                    pt_block = gr.Slider(40, 500, value=180, step=10, label="Correction block (ms)")
                with gr.Row():
                    pt_timing = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Timing correction strength")
                    pt_bpm = gr.Number(value=120.0, minimum=40, maximum=240, label="BPM grid")
                    pt_max_shift = gr.Slider(0.0, 100.0, value=35.0, step=5.0, label="Max timing shift (ms)")
                pt_btn = gr.Button("🎯 Run Neural Pitch + Timing", variant="primary")
                pt_out = gr.File(label="Corrected Vocal WAV")
                pt_status = gr.Markdown("Uses pretrained CREPE pitch tracking through torchcrepe. Start with modest correction strength.")
                pt_btn.click(correct_pitch_timing_with_ai, inputs=[pt_in, pt_fmin, pt_fmax, pt_model, pt_device, pt_periodicity, pt_pitch, pt_max_semitones, pt_block, pt_timing, pt_bpm, pt_max_shift], outputs=[pt_out, pt_status])

            with gr.Tab("🎼 Vocal → Melody / MIDI"):
                melody_in = gr.Audio(label="Vocal Input", type="filepath", sources=["upload", "microphone"])
                with gr.Row():
                    melody_bpm = gr.Slider(40, 240, value=120, step=1, label="BPM")
                    melody_fmin = gr.Number(value=65.41, label="Minimum pitch (Hz)")
                    melody_fmax = gr.Number(value=1046.50, label="Maximum pitch (Hz)")
                melody_btn = gr.Button("🎼 Extract Melody → MIDI", variant="primary")
                melody_out = gr.File(label="MIDI File")
                melody_status = gr.Markdown("Upload a mostly-monophonic vocal, then extract its melody.")
                melody_btn.click(extract_melody, inputs=[melody_in, melody_bpm, melody_fmin, melody_fmax], outputs=[melody_out, melody_status])

            with gr.Tab("🧠 AI MIDI / Basic Pitch"):
                ai_midi_in = gr.Audio(label="Audio Input", type="filepath", sources=["upload", "microphone"])
                with gr.Row():
                    ai_midi_bpm = gr.Slider(40, 240, value=120, step=1, label="MIDI Tempo")
                    ai_min_freq = gr.Number(value=65.41, minimum=0, label="Minimum pitch (0 = automatic)")
                    ai_max_freq = gr.Number(value=1046.50, minimum=0, label="Maximum pitch (0 = automatic)")
                    ai_min_note = gr.Number(value=80, minimum=1, label="Minimum note length (ms)")
                with gr.Row():
                    ai_onset = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Onset threshold")
                    ai_frame = gr.Slider(0.05, 0.95, value=0.3, step=0.05, label="Frame threshold")
                ai_midi_btn = gr.Button("🧠 Transcribe with AI → MIDI", variant="primary")
                ai_midi_out = gr.File(label="AI MIDI File")
                ai_midi_status = gr.Markdown("Optional neural backend. Install the AI requirements in Colab/server first.")
                ai_midi_btn.click(transcribe_ai_midi, inputs=[ai_midi_in, ai_min_freq, ai_max_freq, ai_min_note, ai_onset, ai_frame, ai_midi_bpm], outputs=[ai_midi_out, ai_midi_status])

            with gr.Tab("🤖 AI Conditioned Arrangement"):
                ai_arrange_in = gr.Audio(label="Vocal / Audio Input", type="filepath", sources=["upload", "microphone"])
                gr.Markdown("Estimate tempo, key/scale and vocal activity, then generate synchronized melody + rhythm + chords + bass + drums.")
                with gr.Row():
                    ai_arrange_bpm = gr.Number(value=0, minimum=0, label="BPM override (0 = auto)")
                    ai_arrange_bars = gr.Slider(1, 64, value=8, step=1, label="Bars")
                    ai_arrange_key = gr.Dropdown(["Auto"] + KEYS, value="Auto", label="Key")
                    ai_arrange_scale = gr.Dropdown(["Auto"] + SCALES, value="Auto", label="Scale")
                    ai_arrange_backend = gr.Dropdown(["auto", "basic_pitch", "pyin"], value="auto", label="Melody backend")
                with gr.Row():
                    ai_arrange_min_freq = gr.Number(value=65.41, minimum=0, label="Min pitch (Hz)")
                    ai_arrange_max_freq = gr.Number(value=1046.50, minimum=0, label="Max pitch (Hz)")
                    ai_arrange_min_note = gr.Number(value=80, minimum=1, label="Min note (ms)")
                    ai_arrange_onset = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Onset")
                    ai_arrange_frame = gr.Slider(0.05, 0.95, value=0.3, step=0.05, label="Frame")
                with gr.Row():
                    ai_arrange_density_floor = gr.Slider(0.0, 1.0, value=0.28, step=0.02, label="Density floor")
                    ai_arrange_density_ceiling = gr.Slider(0.0, 1.0, value=0.78, step=0.02, label="Density ceiling")
                    ai_arrange_swing = gr.Slider(-0.5, 0.5, value=0.0, step=0.05, label="Swing")
                    ai_arrange_seed = gr.Number(value=42, precision=0, label="Seed")
                ai_arrange_btn = gr.Button("🤖 Generate AI Arrangement", variant="primary")
                with gr.Row():
                    ai_arrange_out = gr.File(label="AI Arrangement MIDI")
                    ai_arrange_melody_out = gr.File(label="AI Melody MIDI")
                ai_arrange_status = gr.Markdown()
                ai_arrange_btn.click(generate_ai_arrangement, inputs=[ai_arrange_in, ai_arrange_bpm, ai_arrange_bars, ai_arrange_key, ai_arrange_scale, ai_arrange_backend, ai_arrange_min_freq, ai_arrange_max_freq, ai_arrange_min_note, ai_arrange_onset, ai_arrange_frame, ai_arrange_density_floor, ai_arrange_density_ceiling, ai_arrange_swing, ai_arrange_seed], outputs=[ai_arrange_out, ai_arrange_melody_out, ai_arrange_status])

            with gr.Tab("🎵 AI Music Generator"):
                musicgen_in = gr.Audio(label="Vocal / Melody Reference", type="filepath", sources=["upload", "microphone"])
                musicgen_prompt = gr.Textbox(label="Describe the backing music", lines=3, value="Bengali folk-inspired acoustic arrangement, warm harmonium, bamboo flute, hand percussion, soft bass, emotional and organic")
                with gr.Row():
                    musicgen_duration = gr.Slider(2, 30, value=8, step=1, label="Duration (s)")
                    musicgen_guidance = gr.Slider(1, 6, value=3, step=0.1, label="Guidance scale")
                    musicgen_temperature = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Temperature")
                with gr.Row():
                    musicgen_top_k = gr.Slider(0, 500, value=250, step=10, label="Top-k (0 = disabled)")
                    musicgen_top_p = gr.Slider(0, 1, value=0, step=0.05, label="Top-p (0 = disabled)")
                    musicgen_seed = gr.Number(value=42, precision=0, label="Seed")
                    musicgen_device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto", label="Device")
                musicgen_btn = gr.Button("🎵 Generate AI Music", variant="primary")
                musicgen_out = gr.Audio(label="Generated Backing Music", type="filepath")
                musicgen_status = gr.Markdown("MusicGen Melody uses text + audio conditioning. ⚠️ Bundled MusicGen weights are CC-BY-NC 4.0.")
                musicgen_btn.click(generate_musicgen_audio, inputs=[musicgen_in, musicgen_prompt, musicgen_duration, musicgen_guidance, musicgen_temperature, musicgen_top_k, musicgen_top_p, musicgen_seed, musicgen_device], outputs=[musicgen_out, musicgen_status])

            with gr.Tab("🎹 AI Song Builder"):
                gr.Markdown(
                    "## 🎹 Song Sketch Builder\n"
                    "Generate a structured backing-track sketch section-by-section, then crossfade it into one timeline. "
                    "This is a **short sketch workflow (max 90s)** for fast concepting before a full production."
                )
                song_in = gr.Audio(label="Vocal / Melody Reference", type="filepath", sources=["upload", "microphone"])
                song_prompt = gr.Textbox(label="Base style / production prompt", lines=3, value="Bengali folk-inspired acoustic arrangement, warm harmonium, bamboo flute, hand percussion, soft bass, organic emotional production")
                with gr.Row():
                    song_bpm = gr.Number(value=0, minimum=0, maximum=240, label="BPM (0 = no override)")
                    song_key = gr.Dropdown(["Auto"] + KEYS, value="Auto", label="Key")
                    song_scale = gr.Dropdown(["Auto"] + SCALES, value="Auto", label="Scale")
                    song_continuity = gr.Dropdown(["vocal-anchor", "chain"], value="vocal-anchor", label="Continuity mode")
                song_sections = gr.CheckboxGroup(choices=list(SECTION_ORDER), value=list(SECTION_ORDER), label="Song structure")
                with gr.Row():
                    song_intro = gr.Slider(1, 30, value=6, step=1, label="Intro (s)")
                    song_verse = gr.Slider(1, 30, value=8, step=1, label="Verse (s)")
                    song_chorus = gr.Slider(1, 30, value=10, step=1, label="Chorus (s)")
                with gr.Row():
                    song_bridge = gr.Slider(1, 30, value=6, step=1, label="Bridge (s)")
                    song_outro = gr.Slider(1, 30, value=6, step=1, label="Outro (s)")
                    song_crossfade = gr.Slider(0, 2, value=0.45, step=0.05, label="Crossfade (s)")
                with gr.Row():
                    song_guidance = gr.Slider(1, 6, value=3, step=0.1, label="Guidance")
                    song_temperature = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Temperature")
                    song_top_k = gr.Slider(0, 500, value=250, step=10, label="Top-k")
                    song_top_p = gr.Slider(0, 1, value=0, step=0.05, label="Top-p")
                with gr.Row():
                    song_seed = gr.Number(value=42, precision=0, label="Base seed")
                    song_device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto", label="Device")
                song_btn = gr.Button("🎹 Build AI Song Sketch", variant="primary")
                with gr.Row():
                    song_audio_out = gr.Audio(label="Final Song Sketch", type="filepath")
                    song_bundle_out = gr.File(label="Section Bundle (.zip)")
                song_status = gr.Markdown("Each selected section is generated independently and joined with a controlled crossfade.")
                song_btn.click(build_ai_song_sketch, inputs=[song_in, song_prompt, song_bpm, song_key, song_scale, song_sections, song_intro, song_verse, song_chorus, song_bridge, song_outro, song_crossfade, song_guidance, song_temperature, song_top_k, song_top_p, song_seed, song_continuity, song_device], outputs=[song_audio_out, song_bundle_out, song_status])

            with gr.Tab("🎛️ Vocal → Music Parts"):
                parts_in = gr.Audio(label="Vocal / Audio Input", type="filepath", sources=["upload", "microphone"])
                with gr.Row():
                    parts_bpm = gr.Slider(40, 240, value=120, step=1, label="BPM")
                    parts_key = gr.Dropdown(KEYS, value="C", label="Key")
                    parts_scale = gr.Dropdown(SCALES, value="major", label="Scale")
                    parts_bars = gr.Slider(1, 64, value=8, step=1, label="Bars")
                with gr.Row():
                    parts_density = gr.Slider(0.0, 1.0, value=0.55, step=0.05, label="Density")
                    parts_swing = gr.Slider(-0.5, 0.5, value=0.0, step=0.05, label="Swing")
                    parts_seed = gr.Number(value=42, precision=0, label="Seed")
                parts_btn = gr.Button("🎛️ Generate Music Parts", variant="primary")
                with gr.Row():
                    rhythm_out = gr.File(label="Rhythm MIDI")
                    chord_out = gr.File(label="Chord MIDI")
                    bass_out = gr.File(label="Bass MIDI")
                    drum_out = gr.File(label="Drums MIDI")
                parts_status = gr.Markdown()
                parts_btn.click(generate_music_parts, inputs=[parts_in, parts_bpm, parts_key, parts_scale, parts_bars, parts_density, parts_swing, parts_seed], outputs=[rhythm_out, chord_out, bass_out, drum_out, parts_status])

            with gr.Tab("🎼 Vocal → Full Arrangement"):
                arrange_in = gr.Audio(label="Vocal Input", type="filepath", sources=["upload", "microphone"])
                with gr.Row():
                    arrange_bpm = gr.Slider(40, 240, value=120, step=1, label="BPM")
                    arrange_key = gr.Dropdown(KEYS, value="C", label="Key")
                    arrange_scale = gr.Dropdown(SCALES, value="major", label="Scale")
                    arrange_bars = gr.Slider(1, 64, value=8, step=1, label="Bars")
                with gr.Row():
                    arrange_fmin = gr.Number(value=65.41, label="Minimum pitch (Hz)")
                    arrange_fmax = gr.Number(value=1046.50, label="Maximum pitch (Hz)")
                    arrange_seed = gr.Number(value=42, precision=0, label="Seed")
                arrange_btn = gr.Button("🎼 Generate Full MIDI Arrangement", variant="primary")
                with gr.Row():
                    arrange_out = gr.File(label="Arrangement MIDI")
                    arrange_midi_out = gr.File(label="Compatibility MIDI")
                arrange_status = gr.Markdown()
                arrange_btn.click(generate_full_arrangement, inputs=[arrange_in, arrange_bpm, arrange_key, arrange_scale, arrange_bars, arrange_fmin, arrange_fmax, arrange_seed], outputs=[arrange_out, arrange_midi_out, arrange_status])

            with gr.Tab("🎚️ Mix + Master"):
                mix_in = gr.Files(label="Audio Tracks", file_count="multiple", type="filepath")
                with gr.Row():
                    mix_gain = gr.Slider(-12, 12, value=0, step=0.5, label="Master gain (dB)")
                    mix_ratio = gr.Slider(1, 8, value=2, step=0.25, label="Compression ratio")
                    mix_sat = gr.Slider(0, 1, value=0.08, step=0.02, label="Saturation")
                    mix_peak = gr.Slider(0.7, 0.99, value=0.95, step=0.01, label="Target peak")
                mix_btn = gr.Button("🎚️ Mix + Master", variant="primary")
                mix_out = gr.Audio(label="Mastered Mix", type="filepath")
                mix_status = gr.Markdown()
                mix_btn.click(mix_and_master, inputs=[mix_in, mix_gain, mix_ratio, mix_sat, mix_peak], outputs=[mix_out, mix_status])

            with gr.Tab("🧩 Stem Separation"):
                stem_in = gr.Audio(label="Audio Input", type="filepath", sources=["upload", "microphone"])
                with gr.Row():
                    stem_model = gr.Dropdown(["htdemucs"], value="htdemucs", label="Demucs model")
                    stem_shifts = gr.Slider(0, 2, value=1, step=1, label="Quality shifts")
                    stem_overlap = gr.Slider(0.1, 0.75, value=0.25, step=0.05, label="Overlap")
                    stem_device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto", label="Device")
                separate_btn = gr.Button("🧩 Separate Stems", variant="primary")
                stem_outputs = gr.Files(label="Separated WAV Stems")
                stem_status = gr.Markdown("Heavy optional ML feature; GPU/Colab recommended.")
                separate_btn.click(separate_audio, inputs=[stem_in, stem_model, stem_shifts, stem_overlap, stem_device], outputs=[stem_outputs, stem_status])

        gr.Markdown(
            "---\n"
            "### 🧠 Roadmap\n"
            "Current foundation: Vocal Fix DSP · Advanced Vocal Fix · DeepFilterNet · neural pitch/timing · Basic Pitch · "
            "conditioned arrangement · MusicGen audio generation · **AI Song Builder** · MIDI arrangement · mix/master · optional stems.\n\n"
            "Next architecture targets: true time-warping, stronger stem-aware generation, web API, and an original commercially-licensable neural rendering backend."
        )
    return demo


if __name__ == "__main__":
    build_app().launch()
