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


def process_advanced_vocal(path: Optional[str], noise_reduction: float, dereverb_strength: float, pitch_strength: float, timing_strength: float, breath_reduction: float, click_cleanup: bool):
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


def correct_pitch_timing_with_ai(path: Optional[str], fmin: float, fmax: float, model: str, device: str, periodicity: float, pitch_strength: float, max_semitones: float, block_ms: float, timing_strength: float, bpm: float, max_timing_shift_ms: float):
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
            f"Pitch correction: **{float(pitch_strength):.2f}** · Timing correction: **{float(timing_strength):.2f}**\n\n"
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
        midi_path, notes = vocal_to_melody(y, sr, output_path=midi_path, bpm=bpm, config=MelodyConfig(fmin=fmin, fmax=fmax))
        preview = notes[:20]
        lines = [f"### 🎼 Melody extracted — {len(notes)} notes", f"BPM: **{bpm:.0f}**"]
        if preview:
            lines.append("\nFirst notes:")
            lines.extend(f"- MIDI **{n['note']}** · {n['start']:.2f}s · {n['duration']:.2f}s" for n in preview)
        return midi_path, "\n".join(lines)
    except Exception as exc:
        traceback.print_exc()
        return None, f"❌ Melody extraction failed: {type(exc).__name__}: {exc}"


def transcribe_ai_midi(path: Optional[str], min_freq: float, max_freq: float, min_note_ms: float, onset: float, frame: float, bpm: float):
    if not path:
        return None, "Please upload an audio file first."
    try:
        result = transcribe_with_basic_pitch(
            path,
            minimum_frequency=float(min_freq) if min_freq and float(min_freq) > 0 else None,
            maximum_frequency=float(max_freq) if max_freq and float(max_freq) > 0 else None,
            minimum_note_length_ms=max(1.0, float(min_note_ms)),
            onset_threshold=float(onset),
            frame_threshold=float(frame),
            midi_tempo=float(bpm),
        )
        preview = result["notes"][:12]
        lines = ["### 🧠 AI MIDI complete", "Backend: **Spotify Basic Pitch**", f"Notes detected: **{result['note_count']}**", f"MIDI tempo: **{float(bpm):.0f} BPM**"]
        if preview:
            lines.append("\nFirst detected notes:")
            lines.extend(f"- MIDI **{n['note']}** · {n['start']:.2f}s → {n['end']:.2f}s · velocity {n['velocity']:.2f}" for n in preview)
        return result["midi_path"], "\n\n".join(lines)
    except Exception as exc:
        traceback.print_exc()
        return None, f"❌ AI MIDI failed: {type(exc).__name__}: {exc}"


def generate_ai_arrangement(path: Optional[str], bpm_override: float, bars: int, key: str, scale: str, melody_backend: str, min_freq: float, max_freq: float, min_note_ms: float, onset: float, frame: float, density_floor: float, density_ceiling: float, swing: float, seed: int):
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
            f"Tracks: **melody + rhythm + chords + bass + drums**\n\n"
            f"Condition backend: **{result['condition_backend']}**"
        )
        return result["arrangement_path"], result["melody_path"], status
    except Exception as exc:
        traceback.print_exc()
        return None, None, f"❌ AI arrangement failed: {type(exc).__name__}: {exc}"


def generate_musicgen_audio(path: Optional[str], prompt: str, duration: float, guidance: float, temperature: float, top_k: int, top_p: float, device: str):
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


def generate_full_arrangement(path: Optional[str], bpm: float, key: str, scale: str, bars: int, fmin: float, fmax: float, seed: int):
    if not path:
        return None, None, "Please upload a vocal file first."
    try:
        cfg = VocalMusicConfig(bpm=float(bpm), key=str(key), scale=str(scale), bars=int(bars), fmin=float(fmin), fmax=float(fmax), seed=int(seed))
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


def generate_music_parts(path: Optional[str], bpm: float, key: str, scale: str, bars: int, density: float, swing: float, seed: int):
    if not path:
        return None, None, None, None, "Please upload a vocal/audio file first."
    try:
        load_audio(path, mono=True)
        bpm, bars, seed = float(bpm), max(1, int(bars)), int(seed)
        rhythm = generate_rhythm(RhythmConfig(bpm=bpm, bars=bars, density=float(density), swing=float(swing), seed=seed))
        chords = generate_chords(ChordConfig(bpm=bpm, bars=bars, key=str(key), scale=str(scale)))
        bass = generate_bass(BassConfig(bpm=bpm, bars=bars, key=str(key), scale=str(scale)))
        drums = generate_drums(DrumConfig(bpm=bpm, bars=bars, density=float(density), seed=seed))

        def temp_midi(label: str) -> str:
            handle = tempfile.NamedTemporaryFile(suffix=f"_{label.lower().replace(' ', '_')}.mid", delete=False)
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
    if not paths:
        return None, "Please upload at least one audio file."
    try:
        file_paths = list(paths) if isinstance(paths, (list, tuple)) else [paths]
        file_paths = [str(p) for p in file_paths if p]
        if not file_paths:
            return None, "Please upload at least one audio file."
        mixed_path = mix_audio_files(file_paths)
        mixed, sr = load_audio(mixed_path)
        mixed = mixed * (10.0 ** (float(master_gain_db) / 20.0))
        mastered = master_audio(mixed, MasteringConfig(target_peak=float(target_peak), compressor_ratio=float(compression_ratio), saturation=float(saturation)))
        out = save_wav(mastered, sr)
        stats = audio_stats(mastered)
        return out, f"### 🎚️ Mix & Master complete\nTracks mixed: **{len(file_paths)}**\n\nSample rate: **{sr:,} Hz**\n\nPeak: **{stats['peak_dbfs']:.2f} dBFS**\n\nRMS: **{stats['rms_dbfs']:.2f} dBFS**\n\nCrest: **{stats['crest_db']:.2f} dB**"
    except Exception as exc:
        traceback.print_exc()
        return None, f"❌ Mix/Master failed: {type(exc).__name__}: {exc}"


def separate_audio(path: Optional[str], model: str, shifts: int, overlap: float):
    if not path:
        return [], "Please upload an audio file first."
    try:
        outputs = separate_stems(path, config=SeparationConfig(model=model, shifts=int(shifts), overlap=float(overlap)))
        return list(outputs.values()), f"### 🧩 Separation complete\nSources: **{', '.join(outputs.keys())}**\n\nThe files above are the exported WAV stems."
    except Exception as exc:
        traceback.print_exc()
        return [], f"❌ Stem separation failed: {type(exc).__name__}: {exc}"


def build_app():
    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎚️ JE AI Audio Studio\n"
            "### AI-assisted audio editing & music production\n\n"
            "Audio engine + Vocal Fix DSP + Advanced Vocal Fix + Neural Vocal Enhance + Neural Pitch/Timing + AI MIDI + AI Conditioned Arrangement + AI Music Generator + Vocal→MIDI + Full Arrangement + Mix/Master + optional stem separation."
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
                advanced_status = gr.Markdown("Use small pitch/timing values first. The existing Advanced Vocal Fix path is a conservative DSP foundation.")
                advanced_btn.click(process_advanced_vocal, inputs=[advanced_in, advanced_nr, advanced_dereverb, advanced_pitch, advanced_timing, advanced_breath, advanced_click], outputs=[advanced_out, advanced_status])

            with gr.Tab("🧠 Neural Vocal Enhance"):
                neural_in = gr.Audio(label="Vocal / Speech Input", type="filepath", sources=["upload", "microphone"])
                with gr.Row():
                    neural_model = gr.Dropdown(["DeepFilterNet3"], value="DeepFilterNet3", label="Neural Model")
                    neural_device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto", label="Device")
                with gr.Row():
                    neural_atten = gr.Slider(0.0, 30.0, value=12.0, step=1.0, label="Maximum attenuation limit (dB)")
                    neural_post = gr.Checkbox(value=False, label="Post-filter")
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
                    pt_periodicity = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Voicing confidence")
                    pt_pitch = gr.Slider(0.0, 1.0, value=0.65, step=0.05, label="Pitch correction strength")
                    pt_max_semitones = gr.Slider(0.25, 4.0, value=2.0, step=0.25, label="Max pitch move (semitones)")
                with gr.Row():
                    pt_block = gr.Slider(80, 300, value=160, step=10, label="Pitch block (ms)")
                    pt_timing = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Timing correction strength")
                    pt_bpm = gr.Slider(40, 240, value=120, step=1, label="Timing grid BPM")
                    pt_max_shift = gr.Slider(10, 150, value=70, step=5, label="Max timing shift (ms)")
                pt_btn = gr.Button("🎯 Run Neural Pitch + Timing", variant="primary")
                pt_out = gr.File(label="Corrected Vocal WAV")
                pt_status = gr.Markdown("Uses pretrained CREPE pitch tracking through torchcrepe. Start with modest correction strength; the current correction stage is a conservative blockwise processor.")
                pt_btn.click(correct_pitch_timing_with_ai, inputs=[pt_in, pt_fmin, pt_fmax, pt_model, pt_device, pt_periodicity, pt_pitch, pt_max_semitones, pt_block, pt_timing, pt_bpm, pt_max_shift], outputs=[pt_out, pt_status])

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

            with gr.Tab("🧠 AI MIDI / Basic Pitch"):
                ai_midi_in = gr.Audio(label="Audio Input", type="filepath", sources=["upload", "microphone"])
                with gr.Row():
                    ai_midi_bpm = gr.Slider(40, 240, value=120, step=1, label="MIDI Tempo (BPM)")
                    ai_min_freq = gr.Number(value=65.41, minimum=0, label="Minimum pitch (Hz, 0 = automatic)")
                    ai_max_freq = gr.Number(value=1046.50, minimum=0, label="Maximum pitch (Hz, 0 = automatic)")
                with gr.Row():
                    ai_min_note = gr.Slider(20, 1000, value=58, step=1, label="Minimum note length (ms)")
                    ai_onset = gr.Slider(0.05, 0.95, value=0.50, step=0.05, label="Onset threshold")
                    ai_frame = gr.Slider(0.05, 0.95, value=0.30, step=0.05, label="Frame threshold")
                ai_midi_btn = gr.Button("🧠 Transcribe with AI → MIDI", variant="primary")
                ai_midi_out = gr.File(label="AI MIDI File")
                ai_midi_status = gr.Markdown("Optional neural backend. Install the AI requirements in Colab/server before running this tab.")
                ai_midi_btn.click(transcribe_ai_midi, inputs=[ai_midi_in, ai_min_freq, ai_max_freq, ai_min_note, ai_onset, ai_frame, ai_midi_bpm], outputs=[ai_midi_out, ai_midi_status])

            with gr.Tab("🤖 AI Conditioned Arrangement"):
                ai_arrange_in = gr.Audio(label="Vocal / Audio Input", type="filepath", sources=["upload", "microphone"])
                gr.Markdown("Upload a vocal. The conditioning layer estimates tempo, key/scale and vocal activity, then generates synchronized melody + rhythm + chords + bass + drums.")
                with gr.Row():
                    ai_arrange_bpm = gr.Number(value=0, minimum=0, label="BPM override (0 = auto-detect)")
                    ai_arrange_bars = gr.Slider(1, 64, value=8, step=1, label="Bars")
                    ai_arrange_key = gr.Dropdown(["Auto"] + KEYS, value="Auto", label="Key override")
                    ai_arrange_scale = gr.Dropdown(["Auto"] + SCALES, value="Auto", label="Scale override")
                with gr.Row():
                    ai_arrange_backend = gr.Dropdown(["auto", "basic_pitch", "pyin"], value="auto", label="Melody backend")
                    ai_arrange_min_freq = gr.Number(value=65.41, minimum=0, label="Min pitch (Hz)")
                    ai_arrange_max_freq = gr.Number(value=1046.50, minimum=0, label="Max pitch (Hz)")
                with gr.Row():
                    ai_arrange_min_note = gr.Slider(20, 1000, value=58, step=1, label="Min note length (ms)")
                    ai_arrange_onset = gr.Slider(0.05, 0.95, value=0.50, step=0.05, label="Basic Pitch onset")
                    ai_arrange_frame = gr.Slider(0.05, 0.95, value=0.30, step=0.05, label="Basic Pitch frame")
                with gr.Row():
                    ai_arrange_density_floor = gr.Slider(0.0, 1.0, value=0.28, step=0.05, label="Density floor")
                    ai_arrange_density_ceiling = gr.Slider(0.0, 1.0, value=0.78, step=0.05, label="Density ceiling")
                    ai_arrange_swing = gr.Slider(-0.5, 0.5, value=0.0, step=0.05, label="Swing")
                    ai_arrange_seed = gr.Number(value=42, precision=0, label="Seed")
                ai_arrange_btn = gr.Button("🤖 Generate AI Conditioned Arrangement", variant="primary")
                with gr.Row():
                    ai_arrange_out = gr.File(label="AI Arrangement MIDI")
                    ai_arrange_melody_out = gr.File(label="AI Melody MIDI")
                ai_arrange_status = gr.Markdown("This is a modular conditioning pipeline, not yet an end-to-end learned full-song model.")
                ai_arrange_btn.click(generate_ai_arrangement, inputs=[ai_arrange_in, ai_arrange_bpm, ai_arrange_bars, ai_arrange_key, ai_arrange_scale, ai_arrange_backend, ai_arrange_min_freq, ai_arrange_max_freq, ai_arrange_min_note, ai_arrange_onset, ai_arrange_frame, ai_arrange_density_floor, ai_arrange_density_ceiling, ai_arrange_swing, ai_arrange_seed], outputs=[ai_arrange_out, ai_arrange_melody_out, ai_arrange_status])

            with gr.Tab("🎵 AI Music Generator"):
                musicgen_in = gr.Audio(label="Vocal / Melody Reference", type="filepath", sources=["upload", "microphone"])
                musicgen_prompt = gr.Textbox(
                    label="Describe the backing music",
                    lines=3,
                    value="Bengali folk-inspired acoustic arrangement, warm harmonium, bamboo flute, hand percussion, soft bass, emotional and organic",
                )
                with gr.Row():
                    musicgen_duration = gr.Slider(2, 30, value=8, step=1, label="Generation duration (seconds)")
                    musicgen_guidance = gr.Slider(1, 6, value=3, step=0.1, label="Guidance scale")
                    musicgen_temperature = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Temperature")
                with gr.Row():
                    musicgen_top_k = gr.Slider(0, 500, value=250, step=10, label="Top-K")
                    musicgen_top_p = gr.Slider(0, 1, value=0, step=0.05, label="Top-P (0 = disabled)")
                    musicgen_device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto", label="Device")
                musicgen_btn = gr.Button("🎵 Generate AI Music", variant="primary")
                musicgen_out = gr.Audio(label="Generated Music", type="filepath")
                musicgen_status = gr.Markdown(
                    "MusicGen Melody uses both text and an audio/melody reference. The bundled model weights are CC-BY-NC 4.0; use a separately licensed model for commercial deployment."
                )
                musicgen_btn.click(
                    generate_musicgen_audio,
                    inputs=[musicgen_in, musicgen_prompt, musicgen_duration, musicgen_guidance, musicgen_temperature, musicgen_top_k, musicgen_top_p, musicgen_device],
                    outputs=[musicgen_out, musicgen_status],
                )

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
                parts_btn.click(generate_music_parts, inputs=[parts_in, parts_bpm, parts_key, parts_scale, parts_bars, parts_density, parts_swing, parts_seed], outputs=[rhythm_out, chord_out, bass_out, drum_out, parts_status])

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
                arrangement_status = gr.Markdown("Generates melody + rhythm + chords + bass + drums as one synchronized MIDI arrangement.")
                arrange_btn.click(generate_full_arrangement, inputs=[arrange_in, arrange_bpm, arrange_key, arrange_scale, arrange_bars, arrange_fmin, arrange_fmax, arrange_seed], outputs=[arrangement_out, melody_arrangement_out, arrangement_status])

            with gr.Tab("🎚️ Mix & Master"):
                mix_in = gr.Files(label="Upload Stems / Tracks", file_count="multiple", type="filepath")
                with gr.Row():
                    master_gain = gr.Slider(-12, 12, value=0, step=0.5, label="Master Gain (dB)")
                    compression = gr.Slider(1.0, 6.0, value=2.0, step=0.1, label="Compressor Ratio")
                    saturation = gr.Slider(0.0, 0.5, value=0.08, step=0.01, label="Saturation")
                    target_peak = gr.Slider(0.8, 0.99, value=0.95, step=0.01, label="Target Peak")
                mix_btn = gr.Button("🎚️ Mix + Master", variant="primary")
                mix_out = gr.File(label="Mastered WAV")
                mix_status = gr.Markdown("Upload vocals/instruments/stems. The current engine performs a lightweight stereo sum followed by conservative bus processing.")
                mix_btn.click(mix_and_master, inputs=[mix_in, master_gain, compression, saturation, target_peak], outputs=[mix_out, mix_status])

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
            "✅ Audio analysis · ✅ Vocal Fix DSP · ✅ Advanced Vocal Fix · ✅ Neural Vocal Enhance · ✅ Neural Pitch + Timing · ✅ Vocal→Melody/MIDI · ✅ AI MIDI / Basic Pitch · ✅ AI Conditioned Arrangement · ✅ Music Parts · ✅ Stem separation backend · ✅ Full MIDI Arrangement · ✅ Mix/Master foundation · ✅ MusicGen audio-generation backend\n\n"
            "Next: **learned full-song generation/style conditioning → stronger vocal restoration → production web UI/API → Android client.**"
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
