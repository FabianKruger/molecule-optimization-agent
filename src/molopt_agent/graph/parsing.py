"""JSON extraction and SMILES parse/validation graph nodes."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from rdkit import Chem

from ..state import WorkflowState

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _loads_json_object(text: str) -> dict[str, Any]:
    """Parse *text* as a JSON object; raise ValueError/JSONDecodeError otherwise."""
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("JSON payload must be an object")
    return parsed


def _slice_first_json_object(text: str) -> str | None:
    """Return the first top-level `{...}` substring, respecting string literals."""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract a JSON object from model output, tolerating fences and prose."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty model output")

    try:
        return _loads_json_object(text)
    except (json.JSONDecodeError, ValueError):
        pass

    fence = _JSON_FENCE_RE.search(text)
    if fence:
        try:
            return _loads_json_object(fence.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    candidate = _slice_first_json_object(text)
    if candidate is not None:
        return _loads_json_object(candidate)

    raise ValueError("no JSON object found in model output")


def _require_string_field(payload: dict[str, Any], key: str) -> str:
    if key not in payload:
        raise ValueError(f"missing required field '{key}'")
    value = payload[key]
    if value is None:
        raise ValueError(f"field '{key}' must not be null")
    if not isinstance(value, str):
        raise ValueError(f"field '{key}' must be a string")
    return value.strip()


def parse_node(state: WorkflowState) -> WorkflowState:
    iteration = state["iteration_count"]
    raw = state["raw_model_output"]
    try:
        parsed = extract_json_object(raw)
        smiles = _require_string_field(parsed, "smiles")
        reason = _require_string_field(parsed, "reason")
        if not smiles:
            raise ValueError("field 'smiles' must not be empty")
        state["current_smiles"] = smiles
        state["current_reason"] = reason
        state["is_valid"] = True
        state["validation_error"] = ""
        logger.info(f"Iteration {iteration}: Parsed SMILES: {state['current_smiles']}")
    except Exception as e:
        state["current_smiles"] = ""
        state["current_reason"] = ""
        state["is_valid"] = False
        state["validation_error"] = (
            "Invalid JSON. Provide proper JSON with fields 'smiles' and 'reason'."
        )
        logger.error(
            f"Iteration {iteration}: Failed to parse JSON from LLM output: {e}"
        )
    return state


def validation_node(state: WorkflowState) -> WorkflowState:
    iteration = state["iteration_count"]
    smiles = state.get("current_smiles") or ""
    if not isinstance(smiles, str):
        smiles = str(smiles)
    smiles = smiles.strip()
    state["current_smiles"] = smiles

    if not smiles:
        state["is_valid"] = False
        state["validation_error"] = "Invalid SMILES: empty string"
        logger.error(f"Iteration {iteration}: Empty SMILES string")
        return state

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            state["is_valid"] = False
            state["validation_error"] = f"Invalid SMILES: {smiles}"
            logger.error(f"Iteration {iteration}: Invalid SMILES string: {smiles}")
        else:
            state["is_valid"] = True
            state["validation_error"] = ""
            logger.info(f"Iteration {iteration}: SMILES validated successfully")
    except Exception as e:
        state["is_valid"] = False
        state["validation_error"] = f"Validation error: {str(e)}"
        logger.error(f"Iteration {iteration}: SMILES validation error: {e}")
    return state


