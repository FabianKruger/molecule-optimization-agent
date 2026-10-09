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


def test_health_returns_ok() -> None:
    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
    assert health.json() == {"status": "ok"}
