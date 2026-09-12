"""Shared pytest fixtures / env for lean unit tests."""

from __future__ import annotations

import os

# Must run at import time so collection imports see this.
os.environ["TRANSFORMERS_NO_TORCH"] = "1"


def pytest_configure(config):  # noqa: ARG001
    os.environ["TRANSFORMERS_NO_TORCH"] = "1"
