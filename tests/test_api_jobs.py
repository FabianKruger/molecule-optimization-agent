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
from seismo.api.jobs import StreamEvent
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


def test_job_lifecycle_with_fake_stream(jobs_root: Path, api_key: None) -> None:
    del api_key
    gate = threading.Event()
    app = create_app(jobs_root=jobs_root, stream_fn=_fake_stream_factory(gate))

    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200

        created = client.post(
            "/api/jobs",
            json={"preset_id": "qed", "params": {"max_iterations": 3}},
        )
        assert created.status_code == 201
        job_id = created.json()["id"]

        # Second start while running -> 409
        conflict = client.post(
            "/api/jobs",
            json={"preset_id": "qed", "params": {"max_iterations": 3}},
        )
        assert conflict.status_code == 409
        body = conflict.json()
        assert body["detail"] == "A job is already running"
        assert body["running_job_id"] == job_id

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

        # Wait for success.
        for _ in range(50):
            detail = client.get(f"/api/jobs/{job_id}").json()
            if detail["status"] != "running":
                break
            time.sleep(0.05)
        assert detail["status"] == "succeeded"
        assert detail["phase"] is None
        assert detail["summary"] == "Optimization finished."

        listed = client.get("/api/jobs").json()["jobs"]
        assert listed[0]["id"] == job_id
        assert listed[0]["best_score"] == 0.42


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
    app = create_app(
        jobs_root=jobs_root,
        stream_fn=_fake_stream_factory(threading.Event()),
    )
    with TestClient(app) as client:
        detail = client.get(f"/api/jobs/{job_id}").json()
    assert detail["status"] == "failed"
    assert detail["error"] == "interrupted by server restart"
    assert detail["iterations_completed"] == 1
    assert os.environ.get("OPENAI_API_KEY") == "test-key"
