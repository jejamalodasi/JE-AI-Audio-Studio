from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable
import zipfile

import numpy as np

from ai.musicgen_melody import MusicGenConfig, generate_musicgen_melody
from mixing.mixer import mix_audio_arrays
from utils.audio_utils import load_audio, save_wav


SECTION_ORDER = ("Intro", "Verse", "Chorus", "Bridge", "Outro")

_SECTION_GUIDANCE = {
    "Intro": "sparse opening, gentle entrance, establish the tonal mood, leave space for the vocal",
    "Verse": "supportive verse groove, restrained instrumentation, clear rhythmic pocket, leave space for singing",
    "Chorus": "full chorus lift, wider instrumentation, memorable energy, stronger drums and bass, emotional peak",
    "Bridge": "contrasting bridge texture, briefly reduce the groove then build toward the final section",
    "Outro": "warm resolving outro, gradually simpler arrangement, natural ending and gentle release",
}


@dataclass(frozen=True)
class SongSection:
    name: str
    duration_seconds: float


@dataclass(frozen=True)
class SongBuilderConfig:
    """Controls for the section-based MusicGen Song Sketch workflow."""

    base_prompt: str = "Bengali folk-inspired acoustic backing track, warm harmonium, bamboo flute, hand percussion, soft bass, organic emotional production"
    bpm: float | None = None
    key: str | None = None
    scale: str | None = None
    sections: tuple[SongSection, ...] = (
        SongSection("Intro", 6.0),
        SongSection("Verse", 8.0),
        SongSection("Chorus", 10.0),
        SongSection("Bridge", 6.0),
        SongSection("Outro", 6.0),
    )
    crossfade_seconds: float = 0.45
    guidance_scale: float = 3.0
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0
    seed: int = 42
    device: str = "auto"
    continuity: str = "vocal-anchor"
    vocal_gain_db: float = -1.0
    music_gain_db: float = -3.0
    max_total_seconds: float = 90.0


def _sanitize_sections(sections: Iterable[SongSection], max_total_seconds: float) -> list[SongSection]:
    cleaned: list[SongSection] = []
    for section in sections:
        name = str(section.name).strip().title()
        if not name:
            continue
        duration = float(np.clip(float(section.duration_seconds), 1.0, 30.0))
        cleaned.append(SongSection(name, duration))
    if not cleaned:
        raise ValueError("At least one song section is required.")
    total = sum(s.duration_seconds for s in cleaned)
    limit = max(1.0, float(max_total_seconds))
    if total <= limit:
        return cleaned
    scale = limit / total
    return [SongSection(s.name, max(1.0, s.duration_seconds * scale)) for s in cleaned]


def _build_prompt(config: SongBuilderConfig, section: SongSection, index: int, total: int) -> str:
    pieces = [
        str(config.base_prompt).strip(),
        "instrumental backing music only, no lead vocal and no spoken words",
        _SECTION_GUIDANCE.get(section.name, "balanced instrumental section with musical movement"),
        f"section {index + 1} of {total}: {section.name}",
    ]
    if config.bpm and float(config.bpm) > 0:
        pieces.append(f"around {float(config.bpm):.0f} BPM")
    if config.key and str(config.key).lower() != "auto":
        scale = f" {config.scale}" if config.scale and str(config.scale).lower() != "auto" else ""
        pieces.append(f"centered around {config.key}{scale}")
    if config.continuity == "chain":
        pieces.append("continue naturally from the previous section while preserving groove, instrumentation and sonic identity")
    else:
        pieces.append("preserve the reference melody contour and keep instrumentation stylistically consistent with the other sections")
    return ", ".join(pieces)


def _as_stereo(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return np.column_stack([array, array])
    if array.ndim == 2:
        if array.shape[1] == 2:
            return array
        if array.shape[0] == 2 and array.shape[1] != 2:
            return array.T
        return np.column_stack([array[:, 0], array[:, 0]])
    raise ValueError("Generated audio must be mono or stereo.")


def _crossfade_join(chunks: list[tuple[np.ndarray, int]], crossfade_seconds: float) -> tuple[np.ndarray, int]:
    if not chunks:
        raise ValueError("No generated sections were returned.")
    target_sr = chunks[0][1]
    result = _as_stereo(chunks[0][0])
    fade = max(0.0, float(crossfade_seconds))
    for raw_audio, sr in chunks[1:]:
        if int(sr) != target_sr:
            raise ValueError(f"Section sample-rate mismatch: {sr} != {target_sr}")
        nxt = _as_stereo(raw_audio)
        overlap = min(int(round(fade * target_sr)), result.shape[0] // 2, nxt.shape[0] // 2)
        if overlap < 1:
            result = np.concatenate([result, nxt], axis=0)
            continue
        left = np.linspace(1.0, 0.0, overlap, dtype=np.float32)[:, None]
        right = 1.0 - left
        blended = result[-overlap:] * left + nxt[:overlap] * right
        result = np.concatenate([result[:-overlap], blended, nxt[overlap:]], axis=0)
    peak = float(np.max(np.abs(result))) if result.size else 0.0
    if peak > 0.98:
        result *= 0.98 / peak
    return np.clip(result, -1.0, 1.0).astype(np.float32), target_sr


def _prepare_reference(path: str, work_dir: Path) -> str:
    y, sr = load_audio(path, mono=False)
    max_samples = int(30 * sr)
    if y.shape[0] > max_samples:
        y = y[:max_samples]
        return save_wav(y, sr, str(work_dir / "conditioning_reference.wav"))
    return path


def _prepare_vocal_preview(path: str, target_sr: int, target_samples: int, work_dir: Path) -> str:
    y, sr = load_audio(path, mono=False)
    if int(sr) != int(target_sr):
        import librosa
        if y.ndim == 1:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        else:
            y = np.column_stack(
                [librosa.resample(y[:, ch], orig_sr=sr, target_sr=target_sr) for ch in range(y.shape[1])]
            )
    if y.shape[0] > target_samples:
        y = y[:target_samples]
    elif y.shape[0] < target_samples:
        pad_shape = (target_samples - y.shape[0],) + y.shape[1:]
        y = np.concatenate([y, np.zeros(pad_shape, dtype=np.float32)], axis=0)
    return save_wav(y, target_sr, str(work_dir / "vocal_preview.wav"))


def build_song_sketch(
    prompt_audio_path: str,
    config: SongBuilderConfig | None = None,
    output_path: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    if not prompt_audio_path:
        raise ValueError("prompt_audio_path is required")
    source = Path(prompt_audio_path)
    if not source.exists():
        raise FileNotFoundError(str(source))

    cfg = config or SongBuilderConfig()
    sections = _sanitize_sections(cfg.sections, cfg.max_total_seconds)
    total = sum(section.duration_seconds for section in sections)

    generated_sections: list[dict[str, Any]] = []
    chunks: list[tuple[np.ndarray, int]] = []

    persistent_final = Path(output_path) if output_path else source.with_name(f"{source.stem}_ai_song_sketch.wav")
    persistent_preview = persistent_final.with_name(f"{persistent_final.stem}_vocal_preview.wav")
    persistent_bundle = (
        persistent_final.with_suffix(".zip")
        if output_path
        else source.with_name(f"{source.stem}_ai_song_sketch.zip")
    )

    if progress_callback:
        progress_callback(0.0, "starting")

    with tempfile.TemporaryDirectory(prefix="je_song_builder_") as tmp:
        work_dir = Path(tmp)
        reference_path = _prepare_reference(str(source), work_dir)
        chain_reference = reference_path

        for index, section in enumerate(sections):
            if progress_callback:
                progress_callback(
                    5.0 + (index / max(1, len(sections))) * 70.0,
                    f"generating {section.name}",
                )
            section_seed = int(cfg.seed) + index
            prompt = _build_prompt(cfg, section, index, len(sections))
            section_file = work_dir / f"{index + 1:02d}_{section.name.lower()}_musicgen.wav"
            result = generate_musicgen_melody(
                chain_reference,
                prompt,
                output_path=str(section_file),
                config=MusicGenConfig(
                    duration_seconds=section.duration_seconds,
                    guidance_scale=cfg.guidance_scale,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    top_p=cfg.top_p,
                    device=cfg.device,
                    seed=section_seed,
                ),
            )
            audio, sr = load_audio(result["output_path"], mono=False)
            chunks.append((audio, sr))
            generated_sections.append(
                {
                    "name": section.name,
                    "requested_duration_seconds": section.duration_seconds,
                    "actual_duration_seconds": result["duration_seconds"],
                    "prompt": prompt,
                    "seed": section_seed,
                    "filename": section_file.name,
                }
            )
            if cfg.continuity == "chain":
                chain_reference = result["output_path"]

        final_audio, sample_rate = _crossfade_join(chunks, cfg.crossfade_seconds)

        persistent_final.parent.mkdir(parents=True, exist_ok=True)
        save_wav(final_audio, sample_rate, str(persistent_final))

        vocal_path = _prepare_vocal_preview(str(source), sample_rate, final_audio.shape[0], work_dir)
        vocal_audio, _ = load_audio(vocal_path, mono=False)
        full_song = mix_audio_arrays(
            [vocal_audio, final_audio],
            gains_db=[float(cfg.vocal_gain_db), float(cfg.music_gain_db)],
        )
        save_wav(full_song, sample_rate, str(persistent_preview))

        export_dir = work_dir / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        final_export = export_dir / final_file.name
        preview_export = export_dir / preview_file.name
        save_wav(final_audio, sample_rate, str(final_export))
        save_wav(full_song, sample_rate, str(preview_export))

        for index, item in enumerate(generated_sections):
            src = work_dir / f"{index + 1:02d}_{item['name'].lower()}_musicgen.wav"
            dst = export_dir / src.name
            dst.write_bytes(src.read_bytes())

        manifest = {
            "backend": "MusicGen Melody / Transformers",
            "workflow": "JE AI Audio Studio AI Song Builder",
            "total_requested_seconds": total,
            "final_duration_seconds": float(final_audio.shape[0] / sample_rate),
            "sample_rate": sample_rate,
            "crossfade_seconds": cfg.crossfade_seconds,
            "continuity": cfg.continuity,
            "seed": cfg.seed,
            "vocal_gain_db": cfg.vocal_gain_db,
            "music_gain_db": cfg.music_gain_db,
            "sections": generated_sections,
            "license_note": "MusicGen model weights are CC-BY-NC 4.0; use a separately licensed model for commercial deployment.",
        }
        (export_dir / "song_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        bundle_path = persistent_bundle
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for file in export_dir.iterdir():
                bundle.write(file, arcname=file.name)

        final_real_path = str(persistent_final)

    return {
        "output_path": final_real_path,
        "preview_path": str(preview_file),
        "bundle_path": str(bundle_path),
        "sections": generated_sections,
        "section_count": len(generated_sections),
        "requested_duration_seconds": total,
        "duration_seconds": float(final_audio.shape[0] / sample_rate),
        "sampling_rate": sample_rate,
        "backend": "transformers-musicgen-melody-song-builder",
        "continuity": cfg.continuity,
        "license_note": "MusicGen model weights are CC-BY-NC 4.0; use a separately licensed model for commercial deployment.",
    }
