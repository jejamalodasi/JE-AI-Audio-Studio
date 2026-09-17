from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class MusicGenConfig:
    """Controls for optional MusicGen Melody audio synthesis."""

    model_name: str = "facebook/musicgen-melody"
    duration_seconds: float = 8.0
    guidance_scale: float = 3.0
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0
    device: str = "auto"


_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def _device_name(requested: str) -> str:
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if value in {"cpu", "cuda"}:
        return value
    raise ValueError("device must be one of: auto, cpu, cuda")


def _to_mono(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        if array.shape[0] <= 4 and array.shape[1] > array.shape[0]:
            return np.mean(array, axis=0, dtype=np.float32)
        return np.mean(array, axis=1, dtype=np.float32)
    raise ValueError("Audio prompt must be a mono or stereo waveform.")


def _load_model(model_name: str, device: str):
    key = (str(model_name), str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "MusicGen dependencies are not installed. Install the optional AI stack with: "
            "pip install -r requirements-ai.txt"
        ) from exc

    processor = AutoProcessor.from_pretrained(str(model_name))
    model = MusicgenMelodyForConditionalGeneration.from_pretrained(str(model_name))
    model = model.to(device).eval()
    _MODEL_CACHE[key] = (processor, model)
    return processor, model


def generate_musicgen_melody(
    prompt_audio_path: str,
    text_prompt: str,
    output_path: str | None = None,
    config: MusicGenConfig | None = None,
) -> dict[str, Any]:
    """Generate short music audio conditioned by a melody/audio prompt and text.

    The backend is optional because MusicGen model weights are CC-BY-NC 4.0.
    This module is intended for research/prototyping unless a separately
    licensed model is substituted behind the same interface.
    """
    if not prompt_audio_path:
        raise ValueError("prompt_audio_path is required")
    source = Path(prompt_audio_path)
    if not source.exists():
        raise FileNotFoundError(str(source))
    if not text_prompt or not str(text_prompt).strip():
        raise ValueError("text_prompt is required")

    cfg = config or MusicGenConfig()
    duration = float(np.clip(cfg.duration_seconds, 1.0, 30.0))
    temperature = max(0.01, float(cfg.temperature))
    guidance_scale = max(1.0, float(cfg.guidance_scale))
    selected_device = _device_name(cfg.device)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install the optional AI stack with: "
            "pip install -r requirements-ai.txt"
        ) from exc

    from utils.audio_utils import load_audio

    prompt, prompt_sr = load_audio(str(source), mono=False)
    prompt = _to_mono(prompt)
    processor, model = _load_model(cfg.model_name, selected_device)

    inputs = processor(
        audio=prompt,
        sampling_rate=int(prompt_sr),
        text=[str(text_prompt)],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(selected_device) if hasattr(value, "to") else value for key, value in inputs.items()}

    # MusicGen Melody uses an audio-token timeline near 50 steps/sec.
    max_new_tokens = max(50, min(1500, int(round(duration * 50.0))))
    generation_kwargs: dict[str, Any] = {
        "do_sample": True,
        "guidance_scale": guidance_scale,
        "temperature": temperature,
        "top_k": int(max(0, cfg.top_k)),
        "max_new_tokens": max_new_tokens,
    }
    top_p = float(cfg.top_p)
    if top_p > 0.0:
        generation_kwargs["top_p"] = float(np.clip(top_p, 1e-4, 1.0))

    with torch.inference_mode():
        audio_values = model.generate(**inputs, **generation_kwargs)

    audio = audio_values[0].detach().float().cpu().numpy()
    if audio.ndim == 2:
        audio = audio.T
    sampling_rate = int(model.config.audio_encoder.sampling_rate)

    if output_path is None:
        output = source.with_name(f"{source.stem}_musicgen.wav")
    else:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(output), audio, sampling_rate)

    return {
        "output_path": str(output),
        "model": cfg.model_name,
        "device": selected_device,
        "sampling_rate": sampling_rate,
        "duration_seconds": float(audio.shape[-1] / sampling_rate) if audio.size else 0.0,
        "max_new_tokens": max_new_tokens,
        "backend": "transformers-musicgen-melody",
        "license_note": "MusicGen model weights are CC-BY-NC 4.0; use a separately licensed model for commercial deployment.",
    }
