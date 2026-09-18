from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
import zipfile
from typing import Any

import numpy as np

from ai.song_builder import SECTION_ORDER, SongBuilderConfig, SongSection, build_song_sketch
from mixing.mastering import MasteringConfig, master_audio
from mixing.mixer import mix_audio_arrays
from utils.audio_utils import load_audio, save_wav
from vocal.vocal_fix import VocalFixConfig, vocal_fix
from vocal.vocal_fix_advanced import AdvancedVocalFixConfig, advanced_vocal_fix


@dataclass(frozen=True)
class FullSongPipelineConfig:
    """One-click vocal-to-song production pipeline."""

    base_prompt: str = (
        "Bengali folk-inspired acoustic arrangement, warm harmonium, bamboo flute, "
        "hand percussion, soft bass, organic emotional production"
    )
    cleanup_mode: str = "basic"
    bpm: float | None = None
    key: str | None = None
    scale: str | None = None
    sections: tuple[SongSection, ...] = (
        SongSection("Intro", 4.0),
        SongSection("Verse", 8.0),
        SongSection("Chorus", 10.0),
        SongSection("Bridge", 6.0),
        SongSection("Outro", 6.0),
    )
    crossfade_seconds: float = 0.45
    continuity: str = "vocal-anchor"
    guidance_scale: float = 3.0
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0
    seed: int = 42
    vocal_gain_db: float = -1.0
    music_gain_db: float = -3.0
    target_peak: float = 0.95
    compression_ratio: float = 2.0
    saturation: float = 0.08
    device: str = "auto"
    max_total_seconds: float = 90.0


def _resample_audio(y: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if int(source_sr) == int(target_sr):
        return np.asarray(y, dtype=np.float32)
    import librosa

    array = np.asarray(y, dtype=np.float32)
    if array.ndim == 1:
        return librosa.resample(array, orig_sr=source_sr, target_sr=target_sr).astype(np.float32)
    channels = [
        librosa.resample(array[:, ch], orig_sr=source_sr, target_sr=target_sr)
        for ch in range(array.shape[1])
    ]
    return np.column_stack(channels).astype(np.float32)


def _fit_to_length(y: np.ndarray, target_samples: int) -> np.ndarray:
    array = np.asarray(y, dtype=np.float32)
    target_samples = max(1, int(target_samples))
    if array.shape[0] > target_samples:
        return array[:target_samples].copy()
    if array.shape[0] == target_samples:
        return array.copy()
    pad_shape = (target_samples - array.shape[0],) + array.shape[1:]
    return np.concatenate([array, np.zeros(pad_shape, dtype=np.float32)], axis=0)


def _clean_vocal(y: np.ndarray, sr: int, mode: str) -> np.ndarray:
    selected = str(mode or "basic").strip().lower()
    if selected in {"none", "original", "off"}:
        return np.asarray(y, dtype=np.float32)
    if selected in {"basic", "vocal fix", "basic-vocal-fix"}:
        return vocal_fix(
            y,
            sr,
            VocalFixConfig(
                noise_reduction=0.65,
                deesser_threshold=0.22,
                deesser_reduction=0.45,
                compression_ratio=2.0,
                normalize_output=True,
            ),
        )
    if selected in {"advanced", "advanced vocal fix", "advanced-vocal-fix"}:
        return advanced_vocal_fix(
            y,
            sr,
            AdvancedVocalFixConfig(
                noise_reduction=0.65,
                dereverb=0.30,
                pitch_correction=0.0,
                timing_correction=0.0,
                breath_reduction=0.20,
                click_cleanup=True,
            ),
        )
    raise ValueError("cleanup_mode must be one of: none, basic, advanced")


def _copy_file(source: str | Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))
    return str(destination)


def build_full_song_pipeline(
    prompt_audio_path: str,
    config: FullSongPipelineConfig | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run vocal cleanup, AI backing generation, vocal/backing mix, mastering and packaging.

    The AI backing stage uses the optional MusicGen Melody Song Builder. The final
    outputs are ordinary WAV files plus a ZIP bundle with section WAVs and a JSON manifest.
    """
    if not prompt_audio_path:
        raise ValueError("prompt_audio_path is required")

    source = Path(prompt_audio_path)
    if not source.exists():
        raise FileNotFoundError(str(source))

    cfg = config or FullSongPipelineConfig()
    selected_sections = tuple(cfg.sections)
    if not selected_sections:
        raise ValueError("At least one song section is required.")

    with tempfile.TemporaryDirectory(prefix="je_full_song_") as tmp:
        work_dir = Path(tmp)

        source_audio, source_sr = load_audio(str(source), mono=False)
        cleaned = _clean_vocal(source_audio, source_sr, cfg.cleanup_mode)
        cleaned_path = work_dir / "vocal_cleaned.wav"
        save_wav(cleaned, source_sr, str(cleaned_path))

        song_cfg = SongBuilderConfig(
            base_prompt=str(cfg.base_prompt).strip(),
            bpm=cfg.bpm,
            key=cfg.key,
            scale=cfg.scale,
            sections=selected_sections,
            crossfade_seconds=float(cfg.crossfade_seconds),
            guidance_scale=float(cfg.guidance_scale),
            temperature=float(cfg.temperature),
            top_k=int(cfg.top_k),
            top_p=float(cfg.top_p),
            seed=int(cfg.seed),
            device=str(cfg.device),
            continuity=str(cfg.continuity),
            vocal_gain_db=float(cfg.vocal_gain_db),
            music_gain_db=float(cfg.music_gain_db),
            max_total_seconds=float(cfg.max_total_seconds),
        )

        song_result = build_song_sketch(
            str(cleaned_path),
            config=song_cfg,
        )
        backing_audio, backing_sr = load_audio(song_result["output_path"], mono=False)

        vocal_audio = _resample_audio(cleaned, source_sr, backing_sr)
        vocal_audio = _fit_to_length(vocal_audio, backing_audio.shape[0])

        mixed = mix_audio_arrays(
            [vocal_audio, backing_audio],
            gains_db=[float(cfg.vocal_gain_db), float(cfg.music_gain_db)],
        )
        mastered = master_audio(
            mixed,
            MasteringConfig(
                target_peak=float(cfg.target_peak),
                compressor_ratio=float(cfg.compression_ratio),
                makeup_db=1.0,
                saturation=float(cfg.saturation),
            ),
        )

        if output_path:
            final_path = Path(output_path)
            final_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            final_path = source.with_name(f"{source.stem}_je_ai_final_master.wav")
        final_path = final_path.resolve()

        backing_path = final_path.with_name(f"{final_path.stem}_backing.wav")
        vocal_path = final_path.with_name(f"{final_path.stem}_vocal_cleaned.wav")

        save_wav(mastered, backing_sr, str(final_path))
        save_wav(backing_audio, backing_sr, str(backing_path))
        save_wav(vocal_audio, backing_sr, str(vocal_path))

        bundle_path = final_path.with_suffix(".zip")
        manifest = {
            "workflow": "JE AI Audio Studio Full AI Song Pipeline",
            "backend": "MusicGen Melody / Transformers Song Builder",
            "source": source.name,
            "cleanup_mode": cfg.cleanup_mode,
            "bpm": cfg.bpm,
            "key": cfg.key,
            "scale": cfg.scale,
            "continuity": cfg.continuity,
            "seed": int(cfg.seed),
            "sections": [
                {
                    "name": section.name,
                    "duration_seconds": float(section.duration_seconds),
                }
                for section in selected_sections
            ],
            "crossfade_seconds": float(cfg.crossfade_seconds),
            "vocal_gain_db": float(cfg.vocal_gain_db),
            "music_gain_db": float(cfg.music_gain_db),
            "master": {
                "target_peak": float(cfg.target_peak),
                "compression_ratio": float(cfg.compression_ratio),
                "saturation": float(cfg.saturation),
            },
            "final_duration_seconds": float(mastered.shape[0] / backing_sr),
            "sample_rate": int(backing_sr),
            "license_note": (
                "Bundled MusicGen weights are CC-BY-NC 4.0; use a separately licensed "
                "model/backend for commercial deployment."
            ),
        }

        with tempfile.TemporaryDirectory(prefix="je_full_song_bundle_") as bundle_tmp:
            bundle_dir = Path(bundle_tmp)
            _copy_file(final_path, bundle_dir / final_path.name)
            _copy_file(vocal_path, bundle_dir / vocal_path.name)
            _copy_file(backing_path, bundle_dir / backing_path.name)

            for index, section in enumerate(song_result["sections"]):
                section_src = Path(song_result["bundle_path"])
                with zipfile.ZipFile(section_src, "r") as source_zip:
                    member = f"{index + 1:02d}_{section['name'].lower()}_musicgen.wav"
                    try:
                        with source_zip.open(member) as handle, open(bundle_dir / member, "wb") as target:
                            shutil.copyfileobj(handle, target)
                    except KeyError:
                        pass

            (bundle_dir / "song_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for file in bundle_dir.iterdir():
                    bundle.write(file, arcname=file.name)

    return {
        "final_path": str(final_path),
        "vocal_path": str(vocal_path),
        "backing_path": str(backing_path),
        "bundle_path": str(bundle_path),
        "duration_seconds": float(mastered.shape[0] / backing_sr),
        "sample_rate": int(backing_sr),
        "section_count": len(selected_sections),
        "cleanup_mode": cfg.cleanup_mode,
        "continuity": cfg.continuity,
        "seed": int(cfg.seed),
        "backend": "full-song-musicgen-pipeline",
        "license_note": manifest["license_note"],
    }
