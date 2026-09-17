from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PitchTimingAIConfig:
    """Configuration for neural pitch analysis and conservative correction."""

    fmin: float = 65.0
    fmax: float = 1100.0
    hop_ms: float = 10.0
    model: str = "full"
    periodicity_threshold: float = 0.25
    correction_strength: float = 0.65
    max_semitones: float = 2.0
    block_ms: float = 160.0
    block_hop_ms: float = 80.0
    timing_strength: float = 0.0
    bpm: float = 120.0
    max_timing_shift_ms: float = 70.0


def _device_name(requested: str) -> str:
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if value in {"cpu", "cuda", "cuda:0"}:
        return "cuda:0" if value == "cuda" else value
    raise ValueError("device must be one of: auto, cpu, cuda, cuda:0")


def analyze_pitch_neural(
    y: np.ndarray,
    sr: int,
    config: PitchTimingAIConfig | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Estimate frame-wise F0 and periodicity with pretrained CREPE via torchcrepe."""
    cfg = config or PitchTimingAIConfig()
    audio = np.asarray(y, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Audio must be a non-empty mono/stereo waveform.")
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")

    try:
        import torch
        import torchcrepe
    except ImportError as exc:
        raise RuntimeError(
            "Neural pitch tracking is not installed. Install the optional AI dependencies with: "
            "pip install -r requirements-ai.txt"
        ) from exc

    selected_device = _device_name(device)
    hop_length = max(1, int(round(float(sr) * float(cfg.hop_ms) / 1000.0)))
    tensor = torch.from_numpy(audio).unsqueeze(0)

    pitch, periodicity = torchcrepe.predict(
        tensor,
        int(sr),
        hop_length,
        float(cfg.fmin),
        float(cfg.fmax),
        str(cfg.model),
        batch_size=2048,
        device=selected_device,
        return_periodicity=True,
        decoder=torchcrepe.decode.viterbi,
    )

    pitch = pitch.squeeze(0).detach().cpu().numpy().astype(np.float32)
    periodicity = periodicity.squeeze(0).detach().cpu().numpy().astype(np.float32)
    times = (np.arange(pitch.shape[0], dtype=np.float32) * hop_length) / float(sr)
    valid = np.isfinite(pitch) & np.isfinite(periodicity) & (periodicity >= float(cfg.periodicity_threshold))
    pitch = np.where(valid, pitch, 0.0)
    periodicity = np.where(np.isfinite(periodicity), periodicity, 0.0)
    midi = np.where(pitch > 0, 69.0 + 12.0 * np.log2(np.maximum(pitch, 1e-6) / 440.0), 0.0)

    return {
        "pitch_hz": pitch,
        "periodicity": periodicity,
        "midi": midi.astype(np.float32),
        "times": times,
        "hop_length": hop_length,
        "device": selected_device,
        "model": str(cfg.model),
        "backend": "torchcrepe",
    }


def _smoothed_target_shift_semitones(
    pitch_hz: np.ndarray,
    periodicity: np.ndarray,
    sr: int,
    cfg: PitchTimingAIConfig,
) -> np.ndarray:
    pitch = np.asarray(pitch_hz, dtype=np.float32)
    conf = np.asarray(periodicity, dtype=np.float32)
    target = np.zeros_like(pitch)
    voiced = (pitch > 0) & (conf >= float(cfg.periodicity_threshold))
    if not np.any(voiced):
        return target
    midi = np.zeros_like(pitch)
    midi[voiced] = 69.0 + 12.0 * np.log2(np.maximum(pitch[voiced], 1e-6) / 440.0)
    nearest = np.round(midi)
    target[voiced] = nearest[voiced] - midi[voiced]
    target = np.clip(target, -abs(float(cfg.max_semitones)), abs(float(cfg.max_semitones)))
    target *= float(np.clip(cfg.correction_strength, 0.0, 1.0))

    hop_length = max(1, int(round(sr * float(cfg.hop_ms) / 1000.0)))
    win = max(3, int(round(0.06 * sr / hop_length)))
    if win % 2 == 0:
        win += 1
    try:
        from scipy.ndimage import median_filter
        target = median_filter(target, size=win, mode="nearest")
    except ImportError:
        pass
    target[~voiced] = 0.0
    return target.astype(np.float32)


def _pitch_correct_blocks(
    y: np.ndarray,
    sr: int,
    frame_times: np.ndarray,
    frame_shifts: np.ndarray,
    cfg: PitchTimingAIConfig,
) -> np.ndarray:
    import librosa

    audio = np.asarray(y, dtype=np.float32)
    stereo = audio.ndim == 2
    mono = audio if not stereo else np.mean(audio, axis=1)
    block = max(2048, int(round(sr * float(cfg.block_ms) / 1000.0)))
    hop = max(512, int(round(sr * float(cfg.block_hop_ms) / 1000.0)))
    out = np.zeros_like(mono)
    weight = np.zeros_like(mono)

    for start in range(0, len(mono), hop):
        end = min(len(mono), start + block)
        if end - start < 512:
            break
        center = (start + end) / 2.0 / float(sr)
        shift = float(np.interp(
            center,
            frame_times,
            frame_shifts,
            left=float(frame_shifts[0]) if frame_shifts.size else 0.0,
            right=float(frame_shifts[-1]) if frame_shifts.size else 0.0,
        ))
        if abs(shift) < 0.01:
            processed = mono[start:end]
        else:
            processed = librosa.effects.pitch_shift(mono[start:end], sr=sr, n_steps=shift)
        window = np.hanning(len(processed)).astype(np.float32)
        if not np.any(window):
            window = np.ones(len(processed), dtype=np.float32)
        out[start:end] += processed * window
        weight[start:end] += window
        if end >= len(mono):
            break

    valid = weight > 1e-6
    corrected = np.array(mono, copy=True)
    corrected[valid] = out[valid] / weight[valid]
    corrected[~valid] = mono[~valid]
    peak = float(np.max(np.abs(corrected))) if corrected.size else 0.0
    if peak > 0.98:
        corrected *= 0.98 / peak

    if not stereo:
        return corrected.astype(np.float32)
    ratio = np.divide(corrected, mono, out=np.ones_like(corrected), where=np.abs(mono) > 1e-5)
    return (audio * ratio[:, None]).astype(np.float32)


def _note_onsets(midi: np.ndarray, times: np.ndarray, threshold: float = 0.35) -> np.ndarray:
    voiced = np.asarray(midi) > 0
    if not np.any(voiced):
        return np.empty(0, dtype=np.float32)
    onsets: list[float] = []
    prev = 0.0
    for idx, value in enumerate(midi):
        if value <= 0:
            prev = 0.0
            continue
        if prev <= 0 or abs(float(value) - prev) >= float(threshold):
            onsets.append(float(times[idx]))
        prev = float(value)
    return np.asarray(onsets, dtype=np.float32)


def _timing_warp(y: np.ndarray, sr: int, times: np.ndarray, midi: np.ndarray, cfg: PitchTimingAIConfig) -> np.ndarray:
    if cfg.timing_strength <= 0 or cfg.bpm <= 0:
        return np.asarray(y, dtype=np.float32)
    audio = np.asarray(y, dtype=np.float32)
    duration = float(audio.shape[0] / sr)
    onsets = _note_onsets(midi, times)
    if onsets.size < 2:
        return audio.copy()

    beat = 60.0 / float(cfg.bpm)
    max_shift = abs(float(cfg.max_timing_shift_ms)) / 1000.0
    strength = float(np.clip(cfg.timing_strength, 0.0, 1.0))
    anchors_src = [0.0]
    anchors_dst = [0.0]
    last_dst = 0.0
    for onset in onsets:
        target = round(float(onset) / beat) * beat
        delta = np.clip(target - float(onset), -max_shift, max_shift) * strength
        dst = float(onset) + float(delta)
        if dst > last_dst + 0.015 and dst < duration - 0.015:
            anchors_src.append(float(onset))
            anchors_dst.append(dst)
            last_dst = dst
    anchors_src.append(duration)
    anchors_dst.append(duration)
    if len(anchors_src) < 4 or not np.all(np.diff(anchors_dst) > 0):
        return audio.copy()

    target_times = np.arange(audio.shape[0], dtype=np.float64) / float(sr)
    source_times = np.interp(target_times, np.asarray(anchors_dst), np.asarray(anchors_src))
    if audio.ndim == 1:
        return np.interp(source_times, target_times, audio, left=float(audio[0]), right=float(audio[-1])).astype(np.float32)

    channels = [
        np.interp(source_times, target_times, audio[:, ch], left=float(audio[0, ch]), right=float(audio[-1, ch]))
        for ch in range(audio.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def correct_pitch_timing_ai(
    y: np.ndarray,
    sr: int,
    config: PitchTimingAIConfig | None = None,
    device: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run neural frame-wise pitch analysis plus conservative pitch/timing correction."""
    cfg = config or PitchTimingAIConfig()
    analysis = analyze_pitch_neural(y, sr, cfg, device=device)
    corrected = np.asarray(y, dtype=np.float32)

    if cfg.correction_strength > 0:
        shifts = _smoothed_target_shift_semitones(
            analysis["pitch_hz"], analysis["periodicity"], sr, cfg
        )
        corrected = _pitch_correct_blocks(corrected, sr, analysis["times"], shifts, cfg)
    if cfg.timing_strength > 0:
        corrected = _timing_warp(corrected, sr, analysis["times"], analysis["midi"], cfg)

    analysis["voiced_frames"] = int(np.count_nonzero(analysis["pitch_hz"] > 0))
    analysis["note_onsets"] = _note_onsets(analysis["midi"], analysis["times"]).tolist()
    analysis["correction_strength"] = float(cfg.correction_strength)
    analysis["timing_strength"] = float(cfg.timing_strength)
    return np.asarray(corrected, dtype=np.float32), analysis


def correct_pitch_timing_file(
    audio_path: str,
    output_path: str | None = None,
    config: PitchTimingAIConfig | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """File-level wrapper for the neural pitch/timing corrector."""
    source = Path(audio_path)
    if not source.exists():
        raise FileNotFoundError(str(source))
    from utils.audio_utils import load_audio, save_wav

    y, sr = load_audio(str(source))
    corrected, report = correct_pitch_timing_ai(y, sr, config=config, device=device)
    output = Path(output_path) if output_path else source.with_name(f"{source.stem}_ai_pitch_timing.wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_wav(corrected, sr, str(output))
    return {
        "output_path": str(output),
        "sample_rate": int(sr),
        "backend": "torchcrepe",
        "device": report["device"],
        "voiced_frames": report["voiced_frames"],
        "note_onsets": report["note_onsets"],
    }
