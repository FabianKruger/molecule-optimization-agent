"""Tests for SEISMO parse + SMILES validation nodes and JSON extraction helpers."""

from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")

import json

import pytest

rdkit = pytest.importorskip("rdkit")  # noqa: F401
from molopt_agent.graph.parsing import (  # noqa: E402
    extract_json_object,
    parse_node,
    validation_node,
)
from molopt_agent.state import make_initial_state  # noqa: E402


def test_parse_node_accepts_valid_json():
    state = make_initial_state()
    state["iteration_count"] = 1
    state["raw_model_output"] = json.dumps(
        {"reason": "simple alcohol", "smiles": "CCO"}
    )
    out = parse_node(state)
    assert out["is_valid"] is True
    assert out["current_smiles"] == "CCO"
    assert out["current_reason"] == "simple alcohol"
    assert out["validation_error"] == ""


def test_parse_node_rejects_invalid_json():
    state = make_initial_state()
    state["iteration_count"] = 1
    state["raw_model_output"] = "not json"
    out = parse_node(state)
    assert out["is_valid"] is False
    assert out["current_smiles"] == ""
    assert "JSON" in out["validation_error"]


def test_parse_node_extracts_fenced_json():
    state = make_initial_state()
    state["iteration_count"] = 1
    state["raw_model_output"] = (
        "Sure, here you go:\n"
        "```json\n"
        '{"reason": "benzene", "smiles": "c1ccccc1"}\n'
        "```\n"
    )
    out = parse_node(state)
    assert out["is_valid"] is True
    assert out["current_smiles"] == "c1ccccc1"
    assert out["current_reason"] == "benzene"


def test_parse_node_extracts_json_embedded_in_prose():
    state = make_initial_state()
    state["iteration_count"] = 1
    state["raw_model_output"] = (
        'Proposal: {"reason": "ethanol", "smiles": "CCO"} thanks.'
    )
    out = parse_node(state)
    assert out["is_valid"] is True
    assert out["current_smiles"] == "CCO"


def test_parse_node_rejects_missing_fields():
    state = make_initial_state()
    state["iteration_count"] = 1
    state["raw_model_output"] = json.dumps({"smiles": "CCO"})
    out = parse_node(state)
    assert out["is_valid"] is False
    assert out["current_smiles"] == ""


def test_parse_node_rejects_empty_smiles():
    state = make_initial_state()
    state["iteration_count"] = 1
    state["raw_model_output"] = json.dumps({"reason": "empty", "smiles": "  "})
    out = parse_node(state)
    assert out["is_valid"] is False
    assert out["current_smiles"] == ""


def test_parse_node_rejects_non_object_json():
    state = make_initial_state()
    state["iteration_count"] = 1
    state["raw_model_output"] = json.dumps(["CCO"])
    out = parse_node(state)
    assert out["is_valid"] is False


def test_parse_node_strips_field_whitespace():
    state = make_initial_state()
    state["iteration_count"] = 1
    state["raw_model_output"] = json.dumps(
        {"reason": "  alcohol  ", "smiles": "  CCO  "}
    )
    out = parse_node(state)
    assert out["is_valid"] is True
    assert out["current_smiles"] == "CCO"
    assert out["current_reason"] == "alcohol"


def test_extract_json_object_prefers_direct_parse():
    payload = {"reason": "x", "smiles": "C"}
    assert extract_json_object(json.dumps(payload)) == payload


def test_extract_json_object_handles_braces_inside_strings():
    raw = 'prefix {"reason": "has {braces}", "smiles": "CCO"} trailing'
    assert extract_json_object(raw) == {"reason": "has {braces}", "smiles": "CCO"}


def test_validation_node_accepts_valid_smiles():
    state = make_initial_state()
    state["iteration_count"] = 1
    state["current_smiles"] = "c1ccccc1"
    out = validation_node(state)
    assert out["is_valid"] is True
    assert out["validation_error"] == ""


def test_validation_node_rejects_invalid_smiles():
    state = make_initial_state()
    state["iteration_count"] = 1
    state["current_smiles"] = "not_a_smiles"
    out = validation_node(state)
    assert out["is_valid"] is False
    assert "Invalid SMILES" in out["validation_error"]


def test_validation_node_rejects_empty_smiles():
    state = make_initial_state()
    state["iteration_count"] = 1
    state["current_smiles"] = "   "
    out = validation_node(state)
    assert out["is_valid"] is False
    assert "empty" in out["validation_error"].lower()


def test_validation_node_strips_whitespace():
    state = make_initial_state()
    state["iteration_count"] = 1
    state["current_smiles"] = "  CCO  "
    out = validation_node(state)
    assert out["is_valid"] is True
    assert out["current_smiles"] == "CCO"



def test_extract_json_object_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        extract_json_object("   ")


def test_extract_json_object_unclosed_brace():
    with pytest.raises(ValueError):
        extract_json_object('prefix {"reason": "x"')


def test_extract_json_object_skips_bad_fence_then_finds_object():
    raw = "```json\nnot-json\n```\n" + '{"reason": "ok", "smiles": "CCO"}'
    assert extract_json_object(raw) == {"reason": "ok", "smiles": "CCO"}


def test_extract_json_object_handles_escaped_quotes():
    raw = '{"reason": "say \\"hi\\"", "smiles": "CCO"}'
    assert extract_json_object(raw)["reason"] == 'say "hi"'


def test_parse_node_rejects_null_and_non_string_fields():
    state = make_initial_state()
    state["iteration_count"] = 1
    state["raw_model_output"] = json.dumps({"reason": None, "smiles": "CCO"})
    assert parse_node(state)["is_valid"] is False

    state = make_initial_state()
    state["iteration_count"] = 1
    state["raw_model_output"] = json.dumps({"reason": "x", "smiles": 123})
    assert parse_node(state)["is_valid"] is False
