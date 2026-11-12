"""Tests for routing helpers and pure node utilities (no LLM/network)."""

from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from molopt_agent.graph import nodes as node_helpers
from molopt_agent.graph import parsing as parsing_helpers
from molopt_agent.graph.routing import (
    make_route_after_parse,
    make_route_after_prediction,
    make_route_after_validation,
)
from molopt_agent.state import make_initial_state


class _FakeObjective:
    def __init__(self, max_iters: int = 3, done: bool = False):
        self._max_iters = max_iters
        self._done = done

    def max_iterations(self) -> int:
        return self._max_iters

    def is_done(self, state, result) -> bool:
        return self._done


def test_route_after_parse_paths():
    route = make_route_after_parse(_FakeObjective(max_iters=2))

    state = make_initial_state()
    state["iteration_count"] = 2
    assert route(state) == "final"

    state = make_initial_state()
    state["iteration_count"] = 1
    state["is_valid"] = False
    state["current_smiles"] = ""
    assert route(state) == "generation"

    state = make_initial_state()
    state["iteration_count"] = 1
    state["is_valid"] = True
    state["current_smiles"] = "CCO"
    assert route(state) == "validation"


def test_route_after_validation_paths():
    route = make_route_after_validation(_FakeObjective(max_iters=2))

    state = make_initial_state()
    state["iteration_count"] = 2
    assert route(state) == "final"

    state = make_initial_state()
    state["iteration_count"] = 1
    state["is_valid"] = True
    assert route(state) == "prediction"

    state = make_initial_state()
    state["iteration_count"] = 1
    state["is_valid"] = False
    assert route(state) == "generation"


def test_route_after_prediction_paths():
    route_done = make_route_after_prediction(_FakeObjective(max_iters=5, done=True))
    state = make_initial_state()
    state["iteration_count"] = 1
    state["oracle_result"] = {"score": 1.0}
    assert route_done(state) == "final"

    route_max = make_route_after_prediction(_FakeObjective(max_iters=2, done=False))
    state = make_initial_state()
    state["iteration_count"] = 2
    state["oracle_result"] = {"score": 0.1}
    assert route_max(state) == "final"

    route_cont = make_route_after_prediction(_FakeObjective(max_iters=5, done=False))
    state = make_initial_state()
    state["iteration_count"] = 1
    state["oracle_result"] = {"score": 0.1}
    assert route_cont(state) == "generation"


def test_truncate_text():
    assert node_helpers._truncate_text("abc", 10) == "abc"
    assert node_helpers._truncate_text("abcdefghij", 5).endswith("...[truncated]")


def test_message_content_is_blank():
    assert node_helpers._message_content_is_blank("") is True
    assert node_helpers._message_content_is_blank("  ") is True
    assert node_helpers._message_content_is_blank("hi") is False
    assert (
        node_helpers._message_content_is_blank(
            [{"type": "text", "text": "  "}, {"type": "text", "text": ""}]
        )
        is True
    )
    assert (
        node_helpers._message_content_is_blank([{"type": "text", "text": "ok"}])
        is False
    )
    assert node_helpers._message_content_is_blank([{"type": "image"}]) is False


def test_sanitize_messages_drops_blank():
    messages = [
        HumanMessage(content="keep"),
        HumanMessage(content="   "),
        HumanMessage(content="also keep"),
    ]
    cleaned = node_helpers._sanitize_messages(messages)
    assert len(cleaned) == 2
    assert cleaned[0].content == "keep"


def test_serialize_error_body_and_blank_gateway_detection():
    assert node_helpers._serialize_error_body(None) is None
    assert node_helpers._serialize_error_body({"a": 1}) == {"a": 1}
    assert isinstance(node_helpers._serialize_error_body({1, 2}), str)

    err = SimpleNamespace(body={"content": ""})
    assert node_helpers._is_blank_content_gateway_error(err) is True

    err2 = SimpleNamespace(body={"detail": "text field may not be blank"})
    assert node_helpers._is_blank_content_gateway_error(err2) is True

    err3 = SimpleNamespace(body={"ok": True})
    assert node_helpers._is_blank_content_gateway_error(err3) is False


def test_cli_parse_args(monkeypatch: pytest.MonkeyPatch):
    from molopt_agent.cli.args import parse_args

    monkeypatch.setattr(
        "sys.argv",
        ["molopt", "--config", "config/experiments/qed.yaml"],
    )
    args = parse_args()
    assert args.config == "config/experiments/qed.yaml"


def test_make_initial_state_defaults():
    state = make_initial_state()
    assert state["messages"] == []
    assert state["iteration_count"] == 0
    assert state["is_valid"] is False
    assert state["trace"] == []
