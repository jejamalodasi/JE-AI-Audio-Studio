from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class SeparationConfig:
    model: str = "htdemucs"
    shifts: int = 1
    overlap: float = 0.25
    device: str = "auto"


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def separate_stems(
    input_path: str,
    output_dir: Optional[str] = None,
    config: Optional[SeparationConfig] = None,
) -> Dict[str, str]:
    """Separate an audio file with a Demucs-compatible backend.

    The heavy ML dependency is intentionally optional. The main application can
    still start without it; this function gives a clear installation error when
    stem separation is requested without the backend.
    """
    cfg = config or SeparationConfig()
    if not input_path:
        raise ValueError("An input audio file is required.")
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(str(source))

    try:
        from demucs.apply import apply_model
        from demucs.audio import AudioFile, save_audio
        from demucs.pretrained import get_model
    except ImportError as exc:
        raise RuntimeError(
            "Stem separation backend is not installed. Install the optional "
            "'demucs' package in a GPU/Colab environment before using this feature."
        ) from exc

    import torch

    device_name = _resolve_device(cfg.device)
    device = torch.device(device_name)
    model = get_model(cfg.model)
    model.to(device)
    model.eval()

    wav = AudioFile(str(source)).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    wav = wav[None].to(device)

    with torch.no_grad():
        sources = apply_model(
            model,
            wav,
            device=device,
            shifts=max(0, int(cfg.shifts)),
            overlap=float(np.clip(cfg.overlap, 0.0, 0.99)),
            progress=False,
        )

    names = list(model.sources)
    root = Path(output_dir) if output_dir else source.parent / f"{source.stem}_stems"
    root.mkdir(parents=True, exist_ok=True)
    result: Dict[str, str] = {}
    for idx, name in enumerate(names):
        target = root / f"{name}.wav"
        audio = sources[0, idx].detach().cpu()
        save_audio(audio, str(target), samplerate=model.samplerate)
        result[name] = str(target)
    return result
