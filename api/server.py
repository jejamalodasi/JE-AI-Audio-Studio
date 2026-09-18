from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from ai.full_song_pipeline import FullSongPipelineConfig, build_full_song_pipeline
from ai.song_builder import SongSection
from mixing.mastering import MasteringConfig, master_audio
from mixing.mixer import mix_audio_arrays
from music.part_export import build_music_part_midi_bundle
from utils.audio_utils import load_audio, save_wav
from vocal.analyzer import analyze_vocal
from vocal.pitch_correction import PitchCorrectionConfig, correct_pitch
from vocal.timing_correction import TimingCorrectionConfig, correct_timing
from vocal.vocal_fix import VocalFixConfig, vocal_fix


APP_TITLE = "JE AI Audio Studio API"
API_PREFIX = "/api"

_executor = ThreadPoolExecutor(max_workers=max(1, int(os.getenv("JE_AI_API_WORKERS", "1"))))
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _cors_origins() -> list[str]:
    raw = os.getenv("JE_API_CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]


app = FastAPI(
    title=APP_TITLE,
    version="0.1.0",
    description=(
        "HTTP API for the JE AI Audio Studio engine. "
        "Heavy AI generation remains optional and GPU-first."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _job_update(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def _parse_json_config(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="config must be a JSON object")
    return value


def _sections_from_config(value: Any) -> tuple[SongSection, ...]:
    if value is None:
        return (
            SongSection("Intro", 4.0),
            SongSection("Verse", 8.0),
            SongSection("Chorus", 10.0),
            SongSection("Bridge", 6.0),
            SongSection("Outro", 6.0),
        )
    if not isinstance(value, list):
        raise HTTPException(status_code=400, detail="sections must be a JSON list")

    result: list[SongSection] = []
    for item in value:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="each section must be an object")
        name = str(item.get("name", "")).strip().title()
        duration = float(item.get("duration_seconds", 4.0))
        if not name:
            raise HTTPException(status_code=400, detail="section name is required")
        if duration <= 0:
            raise HTTPException(status_code=400, detail="section duration must be positive")
        result.append(SongSection(name, duration))

    if not result:
        raise HTTPException(status_code=400, detail="at least one section is required")
    return tuple(result)


def _build_config(raw: dict[str, Any]) -> FullSongPipelineConfig:
    defaults = FullSongPipelineConfig()
    return FullSongPipelineConfig(
        base_prompt=str(raw.get("base_prompt", defaults.base_prompt)),
        cleanup_mode=str(raw.get("cleanup_mode", defaults.cleanup_mode)),
        bpm=float(raw["bpm"]) if raw.get("bpm") not in (None, "", 0, "0") else None,
        key=None if raw.get("key") in (None, "", "Auto") else str(raw.get("key")),
        scale=None if raw.get("scale") in (None, "", "Auto") else str(raw.get("scale")),
        sections=_sections_from_config(raw.get("sections")),
        crossfade_seconds=float(raw.get("crossfade_seconds", defaults.crossfade_seconds)),
        continuity=str(raw.get("continuity", defaults.continuity)),
        guidance_scale=float(raw.get("guidance_scale", defaults.guidance_scale)),
        temperature=float(raw.get("temperature", defaults.temperature)),
        top_k=int(raw.get("top_k", defaults.top_k)),
        top_p=float(raw.get("top_p", defaults.top_p)),
        seed=int(raw.get("seed", defaults.seed)),
        vocal_gain_db=float(raw.get("vocal_gain_db", defaults.vocal_gain_db)),
        music_gain_db=float(raw.get("music_gain_db", defaults.music_gain_db)),
        target_peak=float(raw.get("target_peak", defaults.target_peak)),
        compression_ratio=float(raw.get("compression_ratio", defaults.compression_ratio)),
        saturation=float(raw.get("saturation", defaults.saturation)),
        device=str(raw.get("device", defaults.device)),
        max_total_seconds=float(raw.get("max_total_seconds", defaults.max_total_seconds)),
    )


def _run_song_job(job_id: str, source_path: Path, output_path: Path, config: FullSongPipelineConfig) -> None:
    def progress(value: float, stage: str) -> None:
        _job_update(
            job_id,
            status="running",
            progress=max(0.0, min(100.0, float(value))),
            stage=str(stage),
        )

    progress(0.0, "starting")
    try:
        result = build_full_song_pipeline(
            str(source_path),
            config=config,
            output_path=str(output_path),
            progress_callback=progress,
        )
        _job_update(
            job_id,
            status="completed",
            progress=100.0,
            stage="done",
            result={
                key: value
                for key, value in result.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
        )
    except Exception as exc:
        _job_update(
            job_id,
            status="failed",
            stage="error",
            error=f"{type(exc).__name__}: {exc}",
        )


def _completed_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "completed" or not job.get("result"):
        raise HTTPException(status_code=409, detail="Job is not completed")
    return job


def _audio_path(job: dict[str, Any], artifact: str) -> Path:
    result = job["result"]
    mapping = {
        "vocal": result.get("vocal_path"),
        "backing": result.get("backing_path"),
        "final": result.get("final_path"),
        "remix": (job.get("remix") or {}).get("final_path"),
    }
    value = mapping.get(artifact)
    if not value:
        raise HTTPException(status_code=404, detail=f"Unknown audio artifact: {artifact}")
    path = Path(value)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return path


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": APP_TITLE,
        "version": "0.1.0",
        "workers": _executor._max_workers,
    }


@app.post(f"{API_PREFIX}/analyze")
async def analyze_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required")

    suffix = Path(file.filename).suffix.lower() or ".wav"
    temp_path = Path(os.getcwd()) / f".je_api_analyze_{uuid.uuid4().hex}{suffix}"
    try:
        temp_path.write_bytes(await file.read())
        y, sr = load_audio(str(temp_path))
        report = analyze_vocal(y, sr)
        return JSONResponse(content=report)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


@app.post(f"{API_PREFIX}/jobs/song")
async def create_song_job(
    file: UploadFile = File(...),
    config: str = Form("{}"),
) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required")

    config_data = _parse_json_config(config)
    pipeline_config = _build_config(config_data)

    job_id = uuid.uuid4().hex
    root = Path(os.getenv("JE_AI_API_JOB_DIR", "/tmp")) / f"je_ai_song_{job_id}"
    root.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix.lower() or ".wav"
    source_path = root / f"source{suffix}"
    output_path = root / "final_master.wav"
    source_path.write_bytes(await file.read())

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "source_filename": file.filename,
        }

    _executor.submit(_run_song_job, job_id, source_path, output_path, pipeline_config)

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "poll": f"{API_PREFIX}/jobs/{job_id}",
        },
    )


@app.get(f"{API_PREFIX}/jobs/{{job_id}}")
def get_song_job(job_id: str) -> JSONResponse:
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = job.get("result")
    if result:
        artifacts = {
            "final": f"{API_PREFIX}/jobs/{job_id}/download/final",
            "vocal": f"{API_PREFIX}/jobs/{job_id}/download/vocal",
            "backing": f"{API_PREFIX}/jobs/{job_id}/download/backing",
            "bundle": f"{API_PREFIX}/jobs/{job_id}/download/bundle",
        }

        remix = job.get("remix") or {}
        if remix.get("final_path"):
            artifacts["remix"] = f"{API_PREFIX}/jobs/{job_id}/download/remix"

        generated_parts = job.get("parts", {}).get("parts", {})
        for part_name, part_path in generated_parts.items():
            if part_path:
                artifacts[part_name] = f"{API_PREFIX}/jobs/{job_id}/download/{part_name}"

        job["artifacts"] = artifacts
        job["clip_edits"] = [
            {
                "edit_id": edit_id,
                "action": edit.get("action"),
                "source": edit.get("source"),
                "download": f"{API_PREFIX}/jobs/{job_id}/clip-edits/{edit_id}",
            }
            for edit_id, edit in (job.get("clip_edits") or {}).items()
        ]

    return JSONResponse(content=job)


@app.get(f"{API_PREFIX}/jobs/{{job_id}}/download/{{artifact}}")
def download_song_artifact(job_id: str, artifact: str) -> FileResponse:
    job = _completed_job(job_id)
    result = job["result"]

    mapping = {
        "final": result.get("final_path"),
        "vocal": result.get("vocal_path"),
        "backing": result.get("backing_path"),
        "bundle": result.get("bundle_path"),
        "remix": (job.get("remix") or {}).get("final_path"),
        **(job.get("parts") or {}).get("parts", {}),
    }

    path_value = mapping.get(artifact)
    if not path_value:
        raise HTTPException(status_code=404, detail="Unknown artifact")

    path = Path(path_value)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found")

    return FileResponse(
        path=str(path),
        filename=path.name,
        media_type=(
            "application/zip"
            if path.suffix.lower() == ".zip"
            else "audio/midi"
            if path.suffix.lower() in {".mid", ".midi"}
            else "audio/wav"
        ),
    )


@app.post(f"{API_PREFIX}/jobs/{{job_id}}/remix")
async def remix_song_job(
    job_id: str,
    vocal_gain_db: float = Form(-1.0),
    backing_gain_db: float = Form(-3.0),
    target_peak: float = Form(0.95),
    compression_ratio: float = Form(2.0),
    saturation: float = Form(0.08),
) -> JSONResponse:
    job = _completed_job(job_id)
    vocal_path = _audio_path(job, "vocal")
    backing_path = _audio_path(job, "backing")

    vocal, vocal_sr = load_audio(str(vocal_path), mono=False)
    backing, backing_sr = load_audio(str(backing_path), mono=False)

    if vocal_sr != backing_sr:
        import librosa

        if np.asarray(vocal).ndim == 1:
            vocal = librosa.resample(vocal, orig_sr=vocal_sr, target_sr=backing_sr)
        else:
            vocal = np.column_stack([
                librosa.resample(vocal[:, ch], orig_sr=vocal_sr, target_sr=backing_sr)
                for ch in range(vocal.shape[1])
            ])

    mixed = mix_audio_arrays(
        [vocal, backing],
        gains_db=[float(vocal_gain_db), float(backing_gain_db)],
    )
    remixed = master_audio(
        mixed,
        MasteringConfig(
            target_peak=float(target_peak),
            compressor_ratio=float(compression_ratio),
            makeup_db=1.0,
            saturation=float(saturation),
        ),
    )

    root = Path(job["result"]["final_path"]).parent / "remix"
    root.mkdir(parents=True, exist_ok=True)
    final_path = root / f"remix_{uuid.uuid4().hex[:10]}.wav"
    save_wav(remixed, backing_sr, str(final_path))

    remix = {
        "final_path": str(final_path),
        "vocal_gain_db": float(vocal_gain_db),
        "backing_gain_db": float(backing_gain_db),
        "target_peak": float(target_peak),
        "compression_ratio": float(compression_ratio),
        "saturation": float(saturation),
    }
    _job_update(job_id, remix=remix)

    return JSONResponse(
        content={
            "job_id": job_id,
            "status": "completed",
            "remix_path": str(final_path),
            "filename": final_path.name,
        }
    )


@app.post(f"{API_PREFIX}/jobs/{{job_id}}/parts")
async def generate_parts(
    job_id: str,
    bpm: float = Form(120.0),
    key: str = Form("C"),
    scale: str = Form("major"),
    bars: int = Form(8),
    seed: int = Form(42),
) -> JSONResponse:
    job = _completed_job(job_id)
    vocal_path = _audio_path(job, "vocal")

    root = Path(job["result"]["final_path"]).parent / "music_parts"
    root.mkdir(parents=True, exist_ok=True)

    try:
        parts = build_music_part_midi_bundle(
            str(vocal_path),
            output_dir=str(root),
            bpm=float(bpm),
            key=str(key or "C"),
            scale=str(scale or "major"),
            bars=max(1, min(128, int(bars))),
            seed=int(seed),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc

    _job_update(job_id, parts=parts)
    return JSONResponse(
        content={
            "job_id": job_id,
            "status": "completed",
            "parts": parts["parts"],
            "metadata": {key: value for key, value in parts.items() if key != "parts"},
        }
    )


@app.post(f"{API_PREFIX}/jobs/{{job_id}}/clip-edit")
async def edit_clip(
    job_id: str,
    source: str = Form("vocal"),
    action: str = Form(...),
    strength: float = Form(0.65),
) -> JSONResponse:
    job = _completed_job(job_id)
    source = str(source).strip().lower()
    action = str(action).strip().lower()

    if source not in {"vocal", "backing"}:
        raise HTTPException(status_code=400, detail="source must be vocal or backing")
    if action not in {"clean", "fix-pitch", "fix-timing"}:
        raise HTTPException(
            status_code=400,
            detail="Supported clip actions are: clean, fix-pitch, fix-timing",
        )
    if source == "backing" and action != "clean":
        raise HTTPException(status_code=400, detail="Pitch/timing correction currently requires a vocal clip")

    source_path = _audio_path(job, source)
    audio, sr = load_audio(str(source_path), mono=False)

    try:
        if action == "clean":
            processed = vocal_fix(
                audio,
                sr,
                VocalFixConfig(
                    noise_reduction=float(np.clip(strength, 0.0, 1.0)),
                    deesser_threshold=0.22,
                    deesser_reduction=0.45,
                    compression_ratio=2.0,
                    normalize_output=True,
                ),
            )
        elif action == "fix-pitch":
            processed = correct_pitch(
                audio,
                sr,
                PitchCorrectionConfig(strength=float(np.clip(strength, 0.0, 1.0))),
            )
        else:
            processed = correct_timing(
                audio,
                sr,
                TimingCorrectionConfig(strength=float(np.clip(strength, 0.0, 1.0))),
            )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc

    edit_id = uuid.uuid4().hex
    root = Path(job["result"]["final_path"]).parent / "clip_edits"
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{edit_id}_{source}_{action}.wav"
    save_wav(processed, sr, str(out))

    with _jobs_lock:
        current = _jobs.setdefault(job_id, {})
        edits = dict(current.get("clip_edits") or {})
        edits[edit_id] = {
            "action": action,
            "source": source,
            "path": str(out),
        }
        current["clip_edits"] = edits

    return JSONResponse(
        content={
            "job_id": job_id,
            "edit_id": edit_id,
            "status": "completed",
            "action": action,
            "source": source,
            "download": f"{API_PREFIX}/jobs/{job_id}/clip-edits/{edit_id}",
        }
    )


@app.get(f"{API_PREFIX}/jobs/{{job_id}}/clip-edits/{{edit_id}}")
def download_clip_edit(job_id: str, edit_id: str) -> FileResponse:
    job = _completed_job(job_id)
    edit = (job.get("clip_edits") or {}).get(edit_id)
    if not edit:
        raise HTTPException(status_code=404, detail="Clip edit not found")

    path = Path(edit["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Clip edit file not found")

    return FileResponse(path=str(path), filename=path.name, media_type="audio/wav")


@app.get(f"{API_PREFIX}/jobs/{{job_id}}/midi/{{part}}")
def get_midi_notes(job_id: str, part: str) -> JSONResponse:
    job = _completed_job(job_id)
    part = str(part).strip().lower()
    allowed = {"melody", "chords", "bass", "drums", "rhythm", "arrangement"}
    if part not in allowed:
        raise HTTPException(status_code=400, detail="Unknown MIDI part")

    parts = (job.get("parts") or {}).get("parts") or {}
    part_path = parts.get(part)
    if not part_path:
        raise HTTPException(status_code=404, detail="MIDI part has not been generated yet")

    midi_path = Path(part_path)
    if not midi_path.exists():
        raise HTTPException(status_code=404, detail="MIDI file not found")

    try:
        import mido

        midi = mido.MidiFile(str(midi_path))
        notes: list[dict[str, Any]] = []
        for track in midi.tracks:
            absolute = 0
            active: dict[tuple[int, int], list[tuple[int, int]]] = {}
            track_name = ""
            for message in track:
                absolute += int(message.time)
                if message.type == "track_name":
                    track_name = str(message.name)
                    continue
                if message.type not in {"note_on", "note_off"}:
                    continue

                note = int(getattr(message, "note", -1))
                channel = int(getattr(message, "channel", 0))
                velocity = int(getattr(message, "velocity", 0))
                key = (channel, note)

                if message.type == "note_on" and velocity > 0:
                    active.setdefault(key, []).append((absolute, velocity))
                    continue

                queued = active.get(key)
                if not queued:
                    continue
                start_tick, start_velocity = queued.pop(0)
                if absolute > start_tick:
                    notes.append(
                        {
                            "note": note,
                            "velocity": start_velocity,
                            "startBeat": float(start_tick / midi.ticks_per_beat),
                            "durationBeat": float((absolute - start_tick) / midi.ticks_per_beat),
                            "trackName": track_name or None,
                        }
                    )
                if not queued:
                    active.pop(key, None)

        notes = [note for note in notes if note["durationBeat"] > 0][:3000]
        notes.sort(key=lambda item: (item["startBeat"], item["note"]))
        return JSONResponse(
            content={
                "part": part,
                "ticksPerBeat": midi.ticks_per_beat,
                "notes": notes,
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc
