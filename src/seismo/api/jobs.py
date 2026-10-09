"""Background job API: start one optimization run and poll its trajectory."""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from seismo.api.presets import (
    Preset,
    build_preset_job,
    get_preset,
    list_presets,
    preview_first_message,
    score_label_for_job,
)
from seismo.config import ExperimentConfig
from seismo.graph.builder import API_KEY_ENV, build_workflow
from seismo.objectives.base import Objective
from seismo.state import make_initial_state
from seismo.ui.mol_utils import smiles_to_image_bytes

PHASE_BY_NODE = {
    "generation": "generating",
    "parse": "generating",
    "validation": "generating",
    "prediction": "scoring",
    "final": "summarizing",
}

MAX_QUEUED_JOBS = 5

StreamEvent = tuple[str, Any]
StreamFn = Callable[
    [Objective, ExperimentConfig, int],
    Iterator[StreamEvent],
]


def _utc_now() -> str:
    # Keep microseconds so rapidly queued jobs have a stable FIFO order.
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def default_stream(
    objective: Objective, cfg: ExperimentConfig, recursion_limit: int
) -> Iterator[StreamEvent]:
    """Stream LangGraph updates/values for a web job (not run_experiment)."""
    workflow = build_workflow(objective, cfg)
    initial = make_initial_state()
    for mode, chunk in workflow.stream(
        initial,
        config=RunnableConfig(recursion_limit=recursion_limit),
        stream_mode=["updates", "values"],
    ):
        if mode == "updates" and isinstance(chunk, dict) and chunk:
            node = next(iter(chunk))
            phase = PHASE_BY_NODE.get(str(node))
            if phase is not None:
                yield ("phase", phase)
        elif mode == "values" and isinstance(chunk, dict):
            yield ("state", chunk)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _iteration_payload(entry: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "iteration": entry["iteration"],
        "smiles": entry["smiles"],
        "reason": entry.get("reason", ""),
        "score": _finite_or_none(entry["score"]),
        "explanation": entry.get("explanation") or "",
    }
    scores = entry.get("scores")
    if scores:
        payload["scores"] = {
            name: _finite_or_none(value) for name, value in scores.items()
        }
    return payload


def _rejected_step(state: dict[str, Any]) -> dict[str, Any] | None:
    """A generated step that never reached the oracle (bad JSON or invalid SMILES)."""
    error = state.get("validation_error") or ""
    iteration = state.get("iteration_count") or 0
    if not error or iteration < 1:
        return None
    return {
        "iteration": iteration,
        "smiles": state.get("current_smiles") or "",
        "reason": state.get("current_reason") or "",
        "score": None,
        "explanation": "",
        "error": error,
    }


def _merge_steps(
    scored: list[dict[str, Any]], rejected: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    by_iteration = {**rejected, **{step["iteration"]: step for step in scored}}
    return [by_iteration[n] for n in sorted(by_iteration)]


def _compute_best(
    iterations: list[dict[str, Any]], *, higher_is_better: bool
) -> dict[str, Any] | None:
    best = None
    for step in iterations:
        if step["score"] is None:
            continue
        if best is None:
            best = step
            continue
        if higher_is_better:
            if step["score"] > best["score"]:
                best = step
        else:
            if step["score"] < best["score"]:
                best = step
    if best is None:
        return None
    return {
        "iteration": best["iteration"],
        "smiles": best["smiles"],
        "score": best["score"],
    }


def _queue_sort_key(job: dict[str, Any]) -> tuple[str, str]:
    return (job.get("created_at", ""), job.get("id", ""))


def _queue_positions(jobs: list[dict[str, Any]]) -> dict[str, int]:
    queued = [job for job in jobs if job.get("status") == "queued"]
    queued.sort(key=_queue_sort_key)
    return {job["id"]: index for index, job in enumerate(queued, start=1)}


def _summary_view(
    job: dict[str, Any], *, queue_position: int | None = None
) -> dict[str, Any]:
    best = job.get("best")
    return {
        "id": job["id"],
        "preset_id": job["preset_id"],
        "preset_title": job["preset_title"],
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "iterations_completed": job["iterations_completed"],
        "max_iterations": job["max_iterations"],
        "best_score": None if best is None else best["score"],
        "best_smiles": None if best is None else best["smiles"],
        "queue_position": queue_position,
    }


def _detail_view(
    job: dict[str, Any], *, queue_position: int | None = None
) -> dict[str, Any]:
    detail = _summary_view(job, queue_position=queue_position)
    detail.update(
        {
            "phase": job.get("phase"),
            "error": job.get("error"),
            "params": job.get("params", {}),
            "higher_is_better": job["higher_is_better"],
            "score_label": job["score_label"],
            "summary": job.get("summary"),
            "best": job.get("best"),
            "iterations": job.get("iterations", []),
            "first_message": job.get("first_message"),
        }
    )
    return detail


class JobManager:
    """One-at-a-time in-process job runner backed by data/web_jobs/{id}/job.json."""

    def __init__(
        self,
        root: Path,
        *,
        stream_fn: StreamFn | None = None,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._stream_fn = stream_fn or default_stream
        self._lock = threading.Lock()
        self._running_id: str | None = None
        self._thread: threading.Thread | None = None
        self._cancel_requested: set[str] = set()

    def mark_interrupted_jobs(self) -> None:
        with self._lock:
            for path in self.root.glob("*/job.json"):
                try:
                    job = _read_json(path)
                except (OSError, json.JSONDecodeError):
                    continue
                if job.get("status") != "running":
                    continue
                job["status"] = "failed"
                job["phase"] = None
                job["error"] = "interrupted by server restart"
                job["updated_at"] = _utc_now()
                _atomic_write_json(path, job)
            self._start_next_locked()

    def running_job_id(self) -> str | None:
        with self._lock:
            if self._running_id is not None:
                return self._running_id
            return self._find_running_on_disk()

    def job_path(self, job_id: str) -> Path:
        return self.root / job_id / "job.json"

    def load_job(self, job_id: str) -> dict[str, Any] | None:
        path = self.job_path(job_id)
        if not path.is_file():
            return None
        return _read_json(path)

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for path in self.root.glob("*/job.json"):
            try:
                jobs.append(_read_json(path))
            except (OSError, json.JSONDecodeError):
                continue
        jobs.sort(key=lambda job: job.get("created_at", ""), reverse=True)
        return jobs

    def start_job(self, preset_id: str, params: dict[str, Any]) -> str:
        api_key = os.environ.get(API_KEY_ENV)
        if api_key is None or api_key == "":
            raise HTTPException(
                status_code=503,
                detail=f"Server is missing {API_KEY_ENV}",
            )
        if get_preset(preset_id) is None:
            raise HTTPException(status_code=404, detail=f"Unknown preset: {preset_id}")

        with self._lock:
            running_id = self._running_id or self._find_running_on_disk()
            if running_id is not None:
                if self._queued_count_locked() >= MAX_QUEUED_JOBS:
                    raise _QueueFull()
                return self._create_queued_job_locked(preset_id, params)

            return self._create_and_run_job_locked(preset_id, params)

    def cancel_job(self, job_id: str) -> None:
        """Cancel a queued job immediately, or ask a running job to stop."""
        with self._lock:
            job = self.load_job(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            status = job.get("status")
            if status == "queued":
                self._update_job(
                    job_id,
                    status="cancelled",
                    phase=None,
                    error=None,
                )
                return
            if status != "running":
                raise _CancelConflict(job_id)
            self._cancel_requested.add(job_id)

    def _cancel_requested_for(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancel_requested

    def _find_running_on_disk(self) -> str | None:
        for path in self.root.glob("*/job.json"):
            try:
                job = _read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("status") == "running":
                job_id = job["id"]
                assert isinstance(job_id, str)
                return job_id
        return None

    def _queued_jobs_sorted(self) -> list[dict[str, Any]]:
        queued: list[dict[str, Any]] = []
        for path in self.root.glob("*/job.json"):
            try:
                job = _read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("status") == "queued":
                queued.append(job)
        queued.sort(key=_queue_sort_key)
        return queued

    def _queued_count_locked(self) -> int:
        return len(self._queued_jobs_sorted())

    def _update_job(self, job_id: str, **changes: Any) -> dict[str, Any]:
        path = self.job_path(job_id)
        job = _read_json(path)
        job.update(changes)
        job["updated_at"] = _utc_now()
        _atomic_write_json(path, job)
        return job

    def _build_job_payload(
        self,
        *,
        job_id: str,
        preset: Preset,
        objective: Objective,
        normalized: dict[str, Any],
        status: str,
    ) -> dict[str, Any]:
        now = _utc_now()
        return {
            "id": job_id,
            "preset_id": preset.id,
            "preset_title": preset.title,
            "status": status,
            "phase": "generating" if status == "running" else None,
            "created_at": now,
            "updated_at": now,
            "iterations_completed": 0,
            "max_iterations": int(normalized["max_iterations"]),
            "error": None,
            "params": normalized,
            "higher_is_better": preset.higher_is_better,
            "score_label": score_label_for_job(preset, normalized),
            "first_message": objective.first_message(),
            "summary": None,
            "best": None,
            "iterations": [],
        }

    def _create_queued_job_locked(self, preset_id: str, params: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            preset, objective, _cfg, normalized = build_preset_job(
                preset_id, params, job_dir
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Unknown preset: {preset_id}"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        job = self._build_job_payload(
            job_id=job_id,
            preset=preset,
            objective=objective,
            normalized=normalized,
            status="queued",
        )
        _atomic_write_json(self.job_path(job_id), job)
        return job_id

    def _create_and_run_job_locked(self, preset_id: str, params: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            preset, objective, cfg, normalized = build_preset_job(
                preset_id, params, job_dir
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Unknown preset: {preset_id}"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        job = self._build_job_payload(
            job_id=job_id,
            preset=preset,
            objective=objective,
            normalized=normalized,
            status="running",
        )
        _atomic_write_json(self.job_path(job_id), job)
        self._launch_thread_locked(job_id, preset, objective, cfg)
        return job_id

    def _launch_thread_locked(
        self,
        job_id: str,
        preset: Preset,
        objective: Objective,
        cfg: ExperimentConfig,
    ) -> None:
        self._running_id = job_id
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, preset, objective, cfg),
            name=f"seismo-job-{job_id}",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def _start_next_locked(self) -> None:
        """Start the oldest queued job if the runner is idle. Caller holds lock."""
        if self._running_id is not None:
            return
        disk_running = self._find_running_on_disk()
        if disk_running is not None:
            self._running_id = disk_running
            return

        while True:
            queued = self._queued_jobs_sorted()
            if not queued:
                return
            job = queued[0]
            job_id = job["id"]
            assert isinstance(job_id, str)

            api_key = os.environ.get(API_KEY_ENV)
            if api_key is None or api_key == "":
                self._update_job(
                    job_id,
                    status="failed",
                    phase=None,
                    error=f"Server is missing {API_KEY_ENV}",
                )
                continue

            try:
                preset, objective, cfg, _normalized = build_preset_job(
                    job["preset_id"],
                    job.get("params") or {},
                    self.root / job_id,
                )
            except (KeyError, ValueError) as exc:
                self._update_job(
                    job_id,
                    status="failed",
                    phase=None,
                    error=str(exc),
                )
                continue

            self._update_job(
                job_id,
                status="running",
                phase="generating",
                error=None,
            )
            self._launch_thread_locked(job_id, preset, objective, cfg)
            return

    def _run_job(
        self,
        job_id: str,
        preset: Preset,
        objective: Objective,
        cfg: ExperimentConfig,
    ) -> None:
        try:
            published_steps = 0
            rejected: dict[int, dict[str, Any]] = {}
            final_state: dict[str, Any] | None = None
            cancelled = False
            for kind, payload in self._stream_fn(objective, cfg, cfg.recursion_limit):
                if self._cancel_requested_for(job_id):
                    cancelled = True
                    break
                if kind == "phase":
                    self._update_job(job_id, phase=payload)
                elif kind == "state":
                    final_state = payload
                    step = _rejected_step(payload)
                    if step is not None:
                        rejected.setdefault(step["iteration"], step)
                    scored = [
                        _iteration_payload(entry)
                        for entry in payload.get("trace") or []
                    ]
                    iterations = _merge_steps(scored, rejected)
                    if len(iterations) > published_steps:
                        best = _compute_best(
                            iterations, higher_is_better=preset.higher_is_better
                        )
                        self._update_job(
                            job_id,
                            iterations=iterations,
                            iterations_completed=len(iterations),
                            best=best,
                        )
                        published_steps = len(iterations)

            if cancelled or self._cancel_requested_for(job_id):
                self._update_job(
                    job_id,
                    status="cancelled",
                    phase=None,
                    error=None,
                )
            else:
                summary = (
                    None if final_state is None else final_state.get("final_response")
                )
                self._update_job(
                    job_id,
                    status="succeeded",
                    phase=None,
                    summary=summary or None,
                    error=None,
                )
        except Exception as exc:  # noqa: BLE001 - surface any worker failure
            self._update_job(
                job_id,
                status="failed",
                phase=None,
                error=str(exc),
            )
        finally:
            with self._lock:
                if self._running_id == job_id:
                    self._running_id = None
                self._thread = None
                self._cancel_requested.discard(job_id)
                self._start_next_locked()


class _QueueFull(Exception):
    pass


class _CancelConflict(Exception):
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(job_id)


class CreateJobBody(BaseModel):
    preset_id: str
    params: dict[str, Any] = Field(default_factory=dict)


class PreviewFirstMessageBody(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)


def _job_manager(request: Request) -> JobManager:
    manager = request.app.state.job_manager
    assert isinstance(manager, JobManager)
    return manager


def create_jobs_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/presets")
    def get_presets() -> dict[str, Any]:
        return {"presets": list_presets()}

    @router.post("/api/presets/{preset_id}/first-message")
    def preview_preset_first_message(
        preset_id: str, body: PreviewFirstMessageBody
    ) -> dict[str, str]:
        if get_preset(preset_id) is None:
            raise HTTPException(status_code=404, detail=f"Unknown preset: {preset_id}")
        try:
            message = preview_first_message(preset_id, body.params)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Unknown preset: {preset_id}"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"first_message": message}

    @router.post("/api/jobs")
    def create_job(body: CreateJobBody, request: Request) -> JSONResponse:
        manager = _job_manager(request)
        try:
            job_id = manager.start_job(body.preset_id, body.params)
        except _QueueFull:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": f"Job queue is full ({MAX_QUEUED_JOBS} waiting)",
                },
            )
        return JSONResponse(status_code=201, content={"id": job_id})

    @router.get("/api/jobs")
    def get_jobs(request: Request) -> dict[str, Any]:
        manager = _job_manager(request)
        jobs = manager.list_jobs()
        positions = _queue_positions(jobs)
        return {
            "jobs": [
                _summary_view(job, queue_position=positions.get(job["id"]))
                for job in jobs
            ]
        }

    @router.get("/api/jobs/{job_id}")
    def get_job(job_id: str, request: Request) -> dict[str, Any]:
        manager = _job_manager(request)
        job = manager.load_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        positions = _queue_positions(manager.list_jobs())
        return _detail_view(job, queue_position=positions.get(job_id))

    @router.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, request: Request) -> JSONResponse:
        manager = _job_manager(request)
        try:
            manager.cancel_job(job_id)
        except _CancelConflict:
            return JSONResponse(
                status_code=409,
                content={"detail": "Only a running or queued job can be cancelled"},
            )
        return JSONResponse(status_code=200, content={"id": job_id})

    @router.get("/api/jobs/{job_id}/iterations/{n}/structure")
    def get_structure(job_id: str, n: int, request: Request) -> Response:
        manager = _job_manager(request)
        job = manager.load_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        step = next(
            (item for item in job.get("iterations", []) if item.get("iteration") == n),
            None,
        )
        if step is None:
            raise HTTPException(status_code=404, detail="Iteration not found")
        png = smiles_to_image_bytes(step["smiles"])
        if png is None:
            raise HTTPException(status_code=404, detail="Structure unavailable")
        return Response(content=png, media_type="image/png")

    return router
