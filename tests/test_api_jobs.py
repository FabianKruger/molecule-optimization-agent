"""Preset catalog and background job API tests.

Uses a fake LangGraph stream so nothing calls OpenAI or Boltz.

    pixi run pytest tests/test_api_jobs.py
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from seismo.api.app import create_app
from seismo.api.jobs import MAX_QUEUED_JOBS, StreamEvent
from seismo.api.presets import ASPIRIN_SMILES, list_presets
from seismo.config import ExperimentConfig
from seismo.objectives.base import Objective

ASPIRIN = ASPIRIN_SMILES


def _fake_stream_factory(
    gate: threading.Event,
) -> Any:
    def fake_stream(
        objective: Objective, cfg: ExperimentConfig, recursion_limit: int
    ) -> Iterator[StreamEvent]:
        del objective, cfg, recursion_limit
        yield ("phase", "generating")
        yield ("state", {"trace": [], "final_response": ""})
        # Pause so the test can observe a running job with a scored step.
        gate.wait(timeout=5)
        yield ("phase", "scoring")
        scored = {
            "trace": [
                {
                    "iteration": 1,
                    "smiles": ASPIRIN,
                    "reason": "start from aspirin",
                    "score": 0.42,
                    "explanation": "demo explanation",
                }
            ],
            "final_response": "",
        }
        yield ("state", scored)
        # Leave the scored step visible while still running.
        time.sleep(0.2)
        yield ("phase", "summarizing")
        finished = {
            "trace": scored["trace"],
            "final_response": "Optimization finished.",
        }
        yield ("state", finished)

    return fake_stream


@pytest.fixture
def jobs_root(tmp_path: Path) -> Path:
    root = tmp_path / "web_jobs"
    root.mkdir()
    return root


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_presets_list_boltz2_first() -> None:
    presets = list_presets()
    assert [p["id"] for p in presets] == [
        "boltz2",
        "qed",
        "similarity",
        "similarity_qed",
        "novel",
        "ic50mpro",
        "ic50mpro_novel",
        "ic50mpro_qed_novel",
    ]
    boltz = presets[0]
    assert boltz["title"] == "Boltz-2 binding"
    assert boltz["higher_is_better"] is True
    field_names = [field["name"] for field in boltz["fields"]]
    assert field_names == [
        "protein_sequence",
        "target_binding_probability",
        "score_mode",
        "additional_information",
        "max_iterations",
    ]


def test_composite_preset_builders_use_components_api(tmp_path: Path) -> None:
    """Presets must construct CompositeOracle with components (post-#23)."""
    from seismo.api.presets import build_preset_job

    for preset_id, params in (
        ("similarity_qed", {"max_iterations": 2}),
        ("ic50mpro_novel", {"max_iterations": 2}),
        ("ic50mpro_qed_novel", {"max_iterations": 2}),
    ):
        preset, objective, cfg, normalized = build_preset_job(
            preset_id, params, tmp_path / preset_id
        )
        assert preset.id == preset_id
        assert objective is not None
        assert cfg.experiment_name == preset_id
        assert normalized["max_iterations"] == 2


def test_preview_first_message_for_qed(jobs_root: Path) -> None:
    app = create_app(
        jobs_root=jobs_root,
        stream_fn=_fake_stream_factory(threading.Event()),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/presets/qed/first-message",
            json={"params": {"target_qed": 0.85, "max_iterations": 5}},
        )
    assert response.status_code == 200
    message = response.json()["first_message"]
    assert "QED" in message
    assert "0.85" in message


def test_preview_first_message_rejects_bad_params(jobs_root: Path) -> None:
    app = create_app(
        jobs_root=jobs_root,
        stream_fn=_fake_stream_factory(threading.Event()),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/presets/similarity/first-message",
            json={
                "params": {"target_smiles": "not-a-smiles", "max_iterations": 5},
            },
        )
    assert response.status_code == 400
    assert "Invalid target SMILES" in response.json()["detail"]


def test_preview_first_message_unknown_preset(jobs_root: Path) -> None:
    app = create_app(
        jobs_root=jobs_root,
        stream_fn=_fake_stream_factory(threading.Event()),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/presets/nope/first-message",
            json={"params": {}},
        )
    assert response.status_code == 404


def test_job_lifecycle_with_fake_stream(jobs_root: Path, api_key: None) -> None:
    del api_key
    gate = threading.Event()
    app = create_app(jobs_root=jobs_root, stream_fn=_fake_stream_factory(gate))

    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200

        created = client.post(
            "/api/jobs",
            json={
                "preset_id": "qed",
                "params": {"target_qed": 0.91, "max_iterations": 3},
            },
        )
        assert created.status_code == 201
        job_id = created.json()["id"]

        early = client.get(f"/api/jobs/{job_id}").json()
        assert early["first_message"] is not None
        assert "0.91" in early["first_message"]
        assert "QED" in early["first_message"]

        # Second start while running -> queued (not 409).
        queued = client.post(
            "/api/jobs",
            json={"preset_id": "qed", "params": {"max_iterations": 3}},
        )
        assert queued.status_code == 201
        queued_id = queued.json()["id"]
        waiting = client.get(f"/api/jobs/{queued_id}").json()
        assert waiting["status"] == "queued"
        assert waiting["queue_position"] == 1
        assert waiting["phase"] is None
        assert waiting["iterations"] == []

        # Let the fake stream publish a scored step, then observe it mid-run.
        gate.set()
        detail = None
        for _ in range(50):
            detail = client.get(f"/api/jobs/{job_id}").json()
            if detail["iterations_completed"] >= 1:
                break
            time.sleep(0.05)
        assert detail is not None
        assert detail["iterations_completed"] == 1
        assert detail["status"] in {"running", "succeeded"}
        assert detail["iterations"][0]["smiles"] == ASPIRIN
        assert detail["best"]["score"] == 0.42
        assert detail["first_message"] == early["first_message"]

        # Wait for first job success; second should start without another POST.
        for _ in range(50):
            detail = client.get(f"/api/jobs/{job_id}").json()
            if detail["status"] != "running":
                break
            time.sleep(0.05)
        assert detail["status"] == "succeeded"
        assert detail["phase"] is None
        assert detail["summary"] == "Optimization finished."

        second = None
        for _ in range(50):
            second = client.get(f"/api/jobs/{queued_id}").json()
            if second["status"] != "queued":
                break
            time.sleep(0.05)
        assert second is not None
        assert second["status"] in {"running", "succeeded"}
        assert second["queue_position"] is None

        listed = client.get("/api/jobs").json()["jobs"]
        by_id = {item["id"]: item for item in listed}
        assert by_id[job_id]["best_score"] == 0.42
        assert by_id[queued_id]["status"] in {"running", "succeeded"}


def _unscored_steps_stream(
    objective: Objective, cfg: ExperimentConfig, recursion_limit: int
) -> Iterator[StreamEvent]:
    del objective, cfg, recursion_limit
    nan_step = {
        "iteration": 2,
        "smiles": ASPIRIN,
        "reason": "oracle fails",
        "score": float("nan"),
        "explanation": "",
        "scores": {"Affinity": float("nan"), "QED": 0.5},
    }
    valid_step = {
        "iteration": 3,
        "smiles": "CCO",
        "reason": "ethanol",
        "score": 0.3,
        "explanation": "",
    }
    yield (
        "state",
        {
            "trace": [],
            "iteration_count": 1,
            "current_smiles": "C1CC",
            "current_reason": "broken ring",
            "validation_error": "Invalid SMILES: C1CC",
        },
    )
    yield ("state", {"trace": [], "iteration_count": 2, "validation_error": ""})
    yield (
        "state",
        {"trace": [nan_step], "iteration_count": 2, "validation_error": ""},
    )
    yield (
        "state",
        {
            "trace": [nan_step, valid_step],
            "iteration_count": 3,
            "validation_error": "",
            "final_response": "done",
        },
    )


def test_unscored_steps_stay_in_trajectory(jobs_root: Path, api_key: None) -> None:
    del api_key
    app = create_app(jobs_root=jobs_root, stream_fn=_unscored_steps_stream)
    with TestClient(app) as client:
        created = client.post(
            "/api/jobs",
            json={"preset_id": "qed", "params": {"max_iterations": 3}},
        )
        job_id = created.json()["id"]
        for _ in range(50):
            response = client.get(f"/api/jobs/{job_id}")
            assert response.status_code == 200
            detail = response.json()
            if detail["status"] != "running":
                break
            time.sleep(0.05)

    assert detail["status"] == "succeeded"
    assert detail["iterations_completed"] == 3
    rejected, nan_scored, valid = detail["iterations"]
    assert rejected["iteration"] == 1
    assert rejected["score"] is None
    assert rejected["smiles"] == "C1CC"
    assert rejected["error"] == "Invalid SMILES: C1CC"
    assert nan_scored["iteration"] == 2
    assert nan_scored["score"] is None
    assert nan_scored["scores"] == {"Affinity": None, "QED": 0.5}
    assert valid["iteration"] == 3
    assert detail["best"] == {"iteration": 3, "smiles": "CCO", "score": 0.3}


def test_queue_rejects_sixth_waiting_job(jobs_root: Path, api_key: None) -> None:
    del api_key
    gate = threading.Event()
    app = create_app(jobs_root=jobs_root, stream_fn=_fake_stream_factory(gate))

    with TestClient(app) as client:
        running = client.post(
            "/api/jobs",
            json={"preset_id": "qed", "params": {"max_iterations": 3}},
        )
        assert running.status_code == 201

        queued_ids: list[str] = []
        for _ in range(MAX_QUEUED_JOBS):
            response = client.post(
                "/api/jobs",
                json={"preset_id": "qed", "params": {"max_iterations": 3}},
            )
            assert response.status_code == 201
            queued_ids.append(response.json()["id"])

        listed = client.get("/api/jobs").json()["jobs"]
        positions = {
            item["id"]: item["queue_position"]
            for item in listed
            if item["status"] == "queued"
        }
        assert positions == {
            job_id: index for index, job_id in enumerate(queued_ids, start=1)
        }

        overflow = client.post(
            "/api/jobs",
            json={"preset_id": "qed", "params": {"max_iterations": 3}},
        )
        assert overflow.status_code == 409
        body = overflow.json()
        assert body["detail"] == f"Job queue is full ({MAX_QUEUED_JOBS} waiting)"
        assert "running_job_id" not in body


def test_cancel_queued_job(jobs_root: Path, api_key: None) -> None:
    del api_key
    gate = threading.Event()
    app = create_app(jobs_root=jobs_root, stream_fn=_fake_stream_factory(gate))

    with TestClient(app) as client:
        running = client.post(
            "/api/jobs",
            json={"preset_id": "qed", "params": {"max_iterations": 3}},
        )
        assert running.status_code == 201

        queued = client.post(
            "/api/jobs",
            json={"preset_id": "qed", "params": {"max_iterations": 3}},
        )
        assert queued.status_code == 201
        queued_id = queued.json()["id"]

        cancelled = client.post(f"/api/jobs/{queued_id}/cancel")
        assert cancelled.status_code == 200
        detail = client.get(f"/api/jobs/{queued_id}").json()
        assert detail["status"] == "cancelled"
        assert detail["queue_position"] is None

        again = client.post(f"/api/jobs/{queued_id}/cancel")
        assert again.status_code == 409


def test_cancel_running_job_frees_slot(jobs_root: Path, api_key: None) -> None:
    del api_key
    gate = threading.Event()
    app = create_app(jobs_root=jobs_root, stream_fn=_fake_stream_factory(gate))

    with TestClient(app) as client:
        created = client.post(
            "/api/jobs",
            json={"preset_id": "qed", "params": {"max_iterations": 3}},
        )
        assert created.status_code == 201
        job_id = created.json()["id"]

        detail = None
        for _ in range(50):
            detail = client.get(f"/api/jobs/{job_id}").json()
            if detail["status"] == "running":
                break
            time.sleep(0.05)
        assert detail is not None
        assert detail["status"] == "running"

        cancelled = client.post(f"/api/jobs/{job_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["id"] == job_id

        # Unblock the fake stream so the runner can notice cancel between steps.
        gate.set()
        for _ in range(50):
            detail = client.get(f"/api/jobs/{job_id}").json()
            if detail["status"] != "running":
                break
            time.sleep(0.05)
        assert detail["status"] == "cancelled"
        assert detail["phase"] is None
        assert detail["error"] is None
        assert (jobs_root / job_id / "job.json").is_file()

        not_running = client.post(f"/api/jobs/{job_id}/cancel")
        assert not_running.status_code == 409

        again = client.post(
            "/api/jobs",
            json={"preset_id": "qed", "params": {"max_iterations": 3}},
        )
        assert again.status_code == 201
        assert again.json()["id"] != job_id


def test_bad_target_smiles_is_400(jobs_root: Path, api_key: None) -> None:
    del api_key
    app = create_app(
        jobs_root=jobs_root,
        stream_fn=_fake_stream_factory(threading.Event()),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            json={
                "preset_id": "similarity",
                "params": {"target_smiles": "not-a-smiles", "max_iterations": 5},
            },
        )
    assert response.status_code == 400
    assert "Invalid target SMILES" in response.json()["detail"]


def test_max_iterations_above_cap_is_400(jobs_root: Path, api_key: None) -> None:
    del api_key
    app = create_app(
        jobs_root=jobs_root,
        stream_fn=_fake_stream_factory(threading.Event()),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            json={"preset_id": "qed", "params": {"max_iterations": 31}},
        )
    assert response.status_code == 400
    assert "max_iterations" in response.json()["detail"]


def test_missing_api_key_is_503(
    jobs_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = create_app(
        jobs_root=jobs_root,
        stream_fn=_fake_stream_factory(threading.Event()),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            json={"preset_id": "qed", "params": {"max_iterations": 3}},
        )
    assert response.status_code == 503


def test_structure_endpoint_returns_png(jobs_root: Path, api_key: None) -> None:
    del api_key
    gate = threading.Event()
    gate.set()
    app = create_app(jobs_root=jobs_root, stream_fn=_fake_stream_factory(gate))
    with TestClient(app) as client:
        created = client.post(
            "/api/jobs",
            json={"preset_id": "qed", "params": {"max_iterations": 3}},
        )
        job_id = created.json()["id"]
        for _ in range(50):
            detail = client.get(f"/api/jobs/{job_id}").json()
            if detail["status"] == "succeeded":
                break
            time.sleep(0.05)
        response = client.get(f"/api/jobs/{job_id}/iterations/1/structure")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_interrupted_running_job_on_startup(jobs_root: Path, api_key: None) -> None:
    del api_key
    job_id = "00000000-0000-0000-0000-000000000001"
    queued_id = "00000000-0000-0000-0000-000000000002"
    job_dir = jobs_root / job_id
    job_dir.mkdir()
    (job_dir / "job.json").write_text(
        """
{
  "id": "00000000-0000-0000-0000-000000000001",
  "preset_id": "qed",
  "preset_title": "QED",
  "status": "running",
  "phase": "scoring",
  "created_at": "2026-01-01T00:00:00Z",
  "updated_at": "2026-01-01T00:00:00Z",
  "iterations_completed": 1,
  "max_iterations": 5,
  "error": null,
  "params": {"max_iterations": 5},
  "higher_is_better": true,
  "score_label": "QED",
  "summary": null,
  "best": {"iteration": 1, "smiles": "CCO", "score": 0.5},
  "iterations": [
    {
      "iteration": 1,
      "smiles": "CCO",
      "reason": "ethanol",
      "score": 0.5,
      "explanation": ""
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    queued_dir = jobs_root / queued_id
    queued_dir.mkdir()
    (queued_dir / "job.json").write_text(
        """
{
  "id": "00000000-0000-0000-0000-000000000002",
  "preset_id": "qed",
  "preset_title": "QED",
  "status": "queued",
  "phase": null,
  "created_at": "2026-01-01T00:00:01Z",
  "updated_at": "2026-01-01T00:00:01Z",
  "iterations_completed": 0,
  "max_iterations": 3,
  "error": null,
  "params": {"max_iterations": 3},
  "higher_is_better": true,
  "score_label": "QED",
  "first_message": "queued first message",
  "summary": null,
  "best": null,
  "iterations": []
}
""".strip(),
        encoding="utf-8",
    )
    gate = threading.Event()
    app = create_app(
        jobs_root=jobs_root,
        stream_fn=_fake_stream_factory(gate),
    )
    with TestClient(app) as client:
        detail = client.get(f"/api/jobs/{job_id}").json()
        assert detail["status"] == "failed"
        assert detail["error"] == "interrupted by server restart"
        assert detail["iterations_completed"] == 1
        assert detail["first_message"] is None

        started = None
        for _ in range(50):
            started = client.get(f"/api/jobs/{queued_id}").json()
            if started["status"] == "running":
                break
            time.sleep(0.05)
        assert started is not None
        assert started["status"] == "running"
        assert started["queue_position"] is None
        assert os.environ.get("OPENAI_API_KEY") == "test-key"
        gate.set()
