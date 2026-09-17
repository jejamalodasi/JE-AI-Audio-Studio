from __future__ import annotations

import tempfile
import traceback
from typing import Optional

import gradio as gr

from ai.vocal_to_melody import MelodyConfig, vocal_to_melody
from ai.vocal_to_music import VocalMusicConfig, generate_from_vocal
from separation.engine import SeparationConfig, separate_stems
from utils.audio_utils import load_audio, normalize, save_wav, trim_audio
from vocal.analyzer import analyze_vocal
from vocal.vocal_fix import VocalFixConfig, vocal_fix


APP_TITLE = "JE AI Audio Studio"


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
            "Audio engine + Vocal Fix DSP + Vocal→MIDI + Full Arrangement + optional stem separation."
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

            with gr.Tab("🎼 Vocal → Full Arrangement"):
                arrange_in = gr.Audio(label="Vocal Input", type="filepath", sources=["upload", "microphone"])
                with gr.Row():
                    arrange_bpm = gr.Slider(40, 240, value=120, step=1, label="BPM")
                    arrange_key = gr.Dropdown(
                        ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
                        value="C",
                        label="Key",
                    )
                    arrange_scale = gr.Dropdown(["major", "minor"], value="major", label="Scale")
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
            "✅ Audio analysis · ✅ Vocal Fix DSP · ✅ Vocal→Melody/MIDI · ✅ Stem separation backend · ✅ Full MIDI Arrangement\n\n"
            "Next: **Vocal→Rhythm/Bass/Drums/Chords controls → Mix/Master → Web/Android production UI.**"
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
