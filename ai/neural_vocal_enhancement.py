from __future__ import annotations

from pathlib import Path
from typing import Any


_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, str]] = {}


def _device_name(requested: str) -> str:
    value = str(requested or "auto").strip().lower()
    if value not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if value != "auto":
        return value
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _init_model(model_name: str, device: str):
    cache_key = (str(model_name), str(device))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        import torch
        from df.enhance import init_df
    except ImportError as exc:
        raise RuntimeError(
            "DeepFilterNet is not installed. Install the optional AI dependencies with "
            "pip install -r requirements-ai.txt"
        ) from exc

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA-capable PyTorch device is available.")

    try:
        initialized = init_df(default_model=str(model_name), config_allow_defaults=True)
    except TypeError:
        initialized = init_df(str(model_name), config_allow_defaults=True)

    if not isinstance(initialized, tuple) or len(initialized) < 3:
        raise RuntimeError("Unexpected DeepFilterNet initialization result.")

    model, df_state, suffix = initialized[:3]
    model = model.to(device=device).eval()
    cached = (model, df_state, str(suffix or ""))
    _MODEL_CACHE[cache_key] = cached
    return cached


def enhance_vocal_neural(
    audio_path: str,
    output_path: str | None = None,
    model_name: str = "DeepFilterNet3",
    device: str = "auto",
    post_filter: bool = False,
    atten_lim_db: float | None = 12.0,
) -> dict[str, Any]:
    """Enhance noisy vocal/speech audio with optional DeepFilterNet inference.

    The model is loaded lazily and cached per model/device pair. Input audio is
    resampled internally to the model sample rate and returned to its original
    sample rate for predictable integration with the rest of the studio.
    """
    if not audio_path:
        raise ValueError("audio_path is required")
    source = Path(audio_path)
    if not source.exists():
        raise FileNotFoundError(str(source))

    selected_device = _device_name(device)
    model, df_state, suffix = _init_model(model_name, selected_device)

    try:
        import numpy as np
        import torch
        from df.enhance import enhance, load_audio, save_audio
        from df.io import resample
    except ImportError as exc:
        raise RuntimeError(
            "DeepFilterNet runtime dependencies are incomplete. Reinstall requirements-ai.txt."
        ) from exc

    # Use the project's normal decoder only to discover the original sample rate.
    from utils.audio_utils import load_audio as studio_load_audio

    _, original_sr = studio_load_audio(str(source), mono=False)
    audio, _ = load_audio(str(source), sr=df_state.sr())
    if hasattr(audio, "to"):
        audio = audio.to(device=selected_device)
    else:
        audio = torch.as_tensor(audio, device=selected_device)

    atten = None if atten_lim_db is None else float(atten_lim_db)
    if atten is not None and atten < 0:
        raise ValueError("atten_lim_db must be >= 0 or None")

    try:
        enhanced = enhance(
            model,
            df_state,
            audio,
            pad=True,
            atten_lim_db=atten,
            post_filter=bool(post_filter),
        )
    except TypeError:
        # Compatibility with versions where post_filter is configured only at init.
        enhanced = enhance(model, df_state, audio, pad=True, atten_lim_db=atten)

    enhanced = enhanced.to("cpu") if hasattr(enhanced, "to") else torch.as_tensor(enhanced)
    enhanced = resample(enhanced, df_state.sr(), int(original_sr))

    if output_path is None:
        output = source.with_name(f"{source.stem}_neural_enhanced.wav")
    else:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

    save_audio(str(output), enhanced, int(original_sr))

    samples = int(enhanced.shape[-1]) if getattr(enhanced, "ndim", 0) else 0
    return {
        "output_path": str(output),
        "input_path": str(source),
        "model": str(model_name),
        "device": selected_device,
        "sample_rate": int(original_sr),
        "duration_seconds": samples / float(original_sr) if original_sr else 0.0,
        "atten_lim_db": atten,
        "post_filter": bool(post_filter),
        "backend": "deepfilternet",
        "suffix": suffix,
    }
