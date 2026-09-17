from __future__ import annotations

import os
import traceback
from typing import Optional

import gradio as gr
import numpy as np

from utils.audio_utils import load_audio, normalize, peak_dbfs, rms_dbfs, save_wav, trim_audio


APP_TITLE = "JE AI Audio Studio"


def _fmt_db(value: float) -> str:
    return f"{value:.2f} dBFS"


def inspect_audio(path: Optional[str]):
    if not path:
        return "No audio loaded."
    try:
        y, sr = load_audio(path)
        channels = 1 if y.ndim == 1 else y.shape[1]
        duration = y.shape[0] / sr
        return (
            f"**Duration:** {duration:.2f}s\n\n"
            f"**Sample rate:** {sr:,} Hz\n\n"
            f"**Channels:** {channels}\n\n"
            f"**Peak:** {_fmt_db(peak_dbfs(y))}\n\n"
            f"**RMS:** {_fmt_db(rms_dbfs(y))}"
        )
    except Exception as exc:
        return f"❌ {type(exc).__name__}: {exc}"


def process_audio(path: Optional[str], operation: str, start: float, end: float):
    if not path:
        return None, "Please upload an audio file first."
    try:
        y, sr = load_audio(path)
        if operation == "Normalize":
            result = normalize(y)
        elif operation == "Trim":
            result = trim_audio(y, sr, start, end if end > 0 else None)
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
            "### AI-assisted audio editing & music production — Phase 1\n\n"
            "This foundation is intentionally lightweight and crash-resistant. "
            "AI generation modules will be added on top of this audio engine."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    label="Input Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                inspect_btn = gr.Button("🔎 Analyze Audio", variant="secondary")
                info = gr.Markdown("Upload a file and press **Analyze Audio**.")

            with gr.Column(scale=1):
                operation = gr.Radio(
                    ["Normalize", "Trim", "Pass-through"],
                    value="Normalize",
                    label="Operation",
                )
                with gr.Row():
                    start = gr.Number(value=0, minimum=0, label="Start (seconds)")
                    end = gr.Number(value=0, minimum=0, label="End (seconds, 0 = file end)")
                process_btn = gr.Button("⚡ Process", variant="primary")
                output = gr.File(label="Processed WAV")
                status = gr.Markdown()

        gr.Markdown(
            "---\n"
            "### 🚧 Coming next\n"
            "Vocal Fix • Noise Reduction • De-reverb • De-esser • Pitch/Timing Fix • "
            "Vocal→MIDI • Vocal→Melody/Rhythm/Bass/Drums/Chords • Stem Separation • "
            "Mix/Master • Full Song Arrangement"
        )

        inspect_btn.click(inspect_audio, inputs=audio_in, outputs=info)
        process_btn.click(
            process_audio,
            inputs=[audio_in, operation, start, end],
            outputs=[output, status],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
