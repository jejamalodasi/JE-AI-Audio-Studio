from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from ai.full_song_pipeline import FullSongPipelineConfig, build_full_song_pipeline
from ai.song_builder import SongSection
from utils.audio_utils import load_audio
from vocal.analyzer import analyze_vocal


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
    _job_update(job_id, status="running", stage="processing")
    try:
        result = build_full_song_pipeline(
            str(source_path),
            config=config,
            output_path=str(output_path),
        )
        _job_update(
            job_id,
            status="completed",
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
        artifact_names = {
            "final": Path(result["final_path"]).name,
            "vocal": Path(result["vocal_path"]).name,
            "backing": Path(result["backing_path"]).name,
            "bundle": Path(result["bundle_path"]).name,
        }
        job["artifacts"] = {
            name: f"{API_PREFIX}/jobs/{job_id}/download/{kind}"
            for kind, name in artifact_names.items()
        }
    return JSONResponse(content=job)


@app.get(f"{API_PREFIX}/jobs/{{job_id}}/download/{{artifact}}")
def download_song_artifact(job_id: str, artifact: str) -> FileResponse:
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {}))

    result = job.get("result")
    if not result:
        raise HTTPException(status_code=409, detail="Job has no completed artifacts")

    mapping = {
        "final": result.get("final_path"),
        "vocal": result.get("vocal_path"),
        "backing": result.get("backing_path"),
        "bundle": result.get("bundle_path"),
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
        media_type="application/zip" if path.suffix == ".zip" else "audio/wav",
    )
