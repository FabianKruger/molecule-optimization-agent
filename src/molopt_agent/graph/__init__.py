"""Graph package.

`build_workflow` is imported lazily so unit tests can load parse/routing helpers
without pulling the LLM/builder stack.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_workflow"]


def __getattr__(name: str) -> Any:
    if name == "build_workflow":
        from .builder import build_workflow

        return build_workflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
