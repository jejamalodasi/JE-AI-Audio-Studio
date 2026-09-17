from __future__ import annotations

import traceback
from typing import Optional

import gradio as gr

from utils.audio_utils import load_audio, normalize, peak_dbfs, rms_dbfs, save_wav, trim_audio
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


def build_app():
    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎚️ JE AI Audio Studio\n"
            "### AI-assisted audio editing & music production\n\n"
            "Phase 1 audio engine + first-generation Vocal Fix DSP pipeline."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(label="Input Audio", type="filepath", sources=["upload", "microphone"])
                inspect_btn = gr.Button("🔎 Analyze Audio", variant="secondary")
                info = gr.Markdown("Upload a file and press **Analyze Audio**.")

            with gr.Column(scale=1):
                operation = gr.Radio(
                    ["Vocal Fix", "Normalize", "Trim", "Pass-through"],
                    value="Vocal Fix",
                    label="Operation",
                )
                nr = gr.Slider(0.0, 2.0, value=0.65, step=0.05, label="Noise Reduction Strength")
                with gr.Row():
                    start = gr.Number(value=0, minimum=0, label="Start (seconds)")
                    end = gr.Number(value=0, minimum=0, label="End (seconds, 0 = file end)")
                process_btn = gr.Button("⚡ Process", variant="primary")
                output = gr.File(label="Processed WAV")
                status = gr.Markdown()

        gr.Markdown(
            "---\n### 🧠 Pipeline status\n"
            "Vocal Fix: **Noise Reduction + High-pass + De-esser + Gentle Compression + Peak Safety**\n\n"
            "Next: ML pitch correction, vocal→MIDI, stem separation, melody/rhythm/bass/drum/chord generation."
        )

        inspect_btn.click(inspect_audio, inputs=audio_in, outputs=info)
        process_btn.click(process_audio, inputs=[audio_in, operation, start, end, nr], outputs=[output, status])

    return demo


if __name__ == "__main__":
    build_app().launch()
