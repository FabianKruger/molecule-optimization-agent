"""Health-check smoke test for the FastAPI shell.

Run from the pixi environment:

    pixi run pytest tests/test_api_health.py
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from seismo.api.app import create_app, parse_args


def test_port_defaults_to_8000() -> None:
    assert parse_args([]).port == 8000


def test_port_flag_overrides_default() -> None:
    assert parse_args(["--port", "9001"]).port == 9001
    assert parse_args(["-p", "9001"]).port == 9001


def test_port_flag_rejects_out_of_range() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--port", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--port", "65536"])


def test_health_returns_ok_with_unknown_build_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SEISMO_BUILD_TIME", raising=False)
    monkeypatch.delenv("SEISMO_BUILD_SHA", raising=False)
    monkeypatch.delenv("SEISMO_DEPLOY_SHA", raising=False)

    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "build_time": "unknown",
        "app_sha": "unknown",
        "deploy_sha": "unknown",
    }


def test_health_returns_build_metadata_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEISMO_BUILD_TIME", "2026-04-01T12:34:56Z")
    monkeypatch.setenv("SEISMO_BUILD_SHA", "abc1234")
    monkeypatch.setenv("SEISMO_DEPLOY_SHA", "def5678")

    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "build_time": "2026-04-01T12:34:56Z",
        "app_sha": "abc1234",
        "deploy_sha": "def5678",
    }


def test_health_treats_blank_build_env_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEISMO_BUILD_TIME", "  ")
    monkeypatch.setenv("SEISMO_BUILD_SHA", "")
    monkeypatch.delenv("SEISMO_DEPLOY_SHA", raising=False)

    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "build_time": "unknown",
        "app_sha": "unknown",
        "deploy_sha": "unknown",
    }


def test_create_app_imports_and_lists_presets() -> None:
    """Regression: deleted oracle modules must not break seismo-api import."""
    from seismo.api.presets import list_presets

    app = create_app()
    client = TestClient(app)
    response = client.get("/api/presets")
    assert response.status_code == 200
    body = response.json()
    assert "presets" in body
    ids = [preset["id"] for preset in body["presets"]]
    assert "boltz2" in ids
    assert "opioid_ki" not in ids
    assert ids == [preset["id"] for preset in list_presets()]


def test_frontend_dist_is_served_when_present() -> None:
    from pathlib import Path

    from seismo.api import app as app_module

    dist = Path(app_module.__file__).resolve().parents[3] / "frontend" / "dist"
    index = dist / "index.html"
    if not index.is_file():
        pytest.skip("frontend/dist not built")

    client = TestClient(create_app())
    root = client.get("/")
    assert root.status_code == 200
    assert "text/html" in root.headers["content-type"]
    assert "seismo" in root.text.lower() or "root" in root.text

    spa = client.get("/jobs/example")
    assert spa.status_code == 200
    assert "text/html" in spa.headers["content-type"]

    health = client.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert "build_time" in body
    assert "app_sha" in body
    assert "deploy_sha" in body
