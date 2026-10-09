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
