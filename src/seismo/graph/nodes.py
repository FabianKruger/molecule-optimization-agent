import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Any

import httpcore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError, RateLimitError
from rdkit import Chem

from ..objectives.base import Objective
from ..state import WorkflowState

logger = logging.getLogger(__name__)


def _truncate_text(value: str, max_len: int = 800) -> str:
    if len(value) <= max_len:
        return value
    return value[:max_len] + " ...[truncated]"


def _message_preview(messages: Any, max_items: int = 5) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    if not isinstance(messages, list):
        return previews

    for msg in messages[-max_items:]:
        role = getattr(msg, "type", msg.__class__.__name__)
        content = getattr(msg, "content", "")

        if isinstance(content, str):
            content_text = content
        else:
            content_text = str(content)

        previews.append(
            {
                "role": role,
                "content_len": len(content_text),
                "content_preview": _truncate_text(content_text, 300),
            }
        )

    return previews


def _serialize_error_body(body: Any) -> Any:
    if body is None:
        return None
    try:
        json.dumps(body)
        return body
    except TypeError:
        return str(body)


def _message_content_is_blank(content: Any) -> bool:
    if isinstance(content, str):
        return content.strip() == ""

    if isinstance(content, list):
        # For list-structured content, treat as blank only if all text fragments are blank.
        saw_text = False
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                saw_text = True
                if str(item.get("text", "")).strip() != "":
                    return False
        return saw_text

    return False


def _sanitize_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    return [m for m in messages if not _message_content_is_blank(getattr(m, "content", None))]


def _is_blank_content_gateway_error(error: APIStatusError) -> bool:
    text = str(_serialize_error_body(getattr(error, "body", None)))
    return (
        "content': ''" in text
        or 'content": ""' in text
        or ("text field" in text and "blank" in text)
    )


def _write_api_error_log(
    error: APIStatusError,
    messages: Any,
    attempt: int,
    max_attempts: int,
) -> str:
    os.makedirs("run_logs", exist_ok=True)
    log_path = os.path.join("run_logs", "gateway_api_errors.jsonl")

    response_text = ""
    try:
        response_text = error.response.text
    except Exception:
        response_text = "<unavailable>"

    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": attempt + 1,
        "max_attempts": max_attempts,
        "status_code": error.status_code,
        "request_id": getattr(error, "request_id", None),
        "request_method": error.request.method,
        "request_url": str(error.request.url),
        "error_message": str(error),
        "error_body": _serialize_error_body(getattr(error, "body", None)),
        "response_text": response_text,
        "message_count": len(messages) if isinstance(messages, list) else None,
        "message_preview": _message_preview(messages),
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")

    return log_path


def rate_limit_sensible_llm_call(llm: ChatOpenAI, message, max_attempts=3):
    delay = 60  # sensible default for hard limits
    conn_delay = 0.5

    for attempt in range(max_attempts):
        try:
            sanitized_messages = _sanitize_messages(message)
            if isinstance(message, list) and isinstance(sanitized_messages, list):
                removed = len(message) - len(sanitized_messages)
                if removed > 0:
                    logger.warning(
                        "Removed %s blank messages before LLM invoke", removed
                    )
            return llm.invoke(sanitized_messages)

        except RateLimitError:
            if attempt == max_attempts - 1:
                raise

            print(f"... Wait {delay}s to not exceed rate limit")
            time.sleep(delay)
            delay = min(delay * 2, 300)  # cap at 10s

        except (APIConnectionError, httpcore.RemoteProtocolError) as e:
            if attempt == max_attempts - 1:
                raise

            # exponential backoff + jitter, capped
            wait = min(conn_delay, 15.0) * random.uniform(0.7, 1.3)
            print(f"... Wait {wait} as the connection was ended")
            now = time.time()
            with open("timing_gpt.log", "a", encoding="utf-8") as f:
                f.write(
                    f"Attempt {attempt}: Will wait for {wait} and retry. Connection ended at |{now}\n"
                )
            time.sleep(wait)
            conn_delay = min(conn_delay * 2, 15.0)

        except APIStatusError as e:
            log_path = _write_api_error_log(
                error=e,
                messages=message,
                attempt=attempt,
                max_attempts=max_attempts,
            )
            if e.status_code == 400 and _is_blank_content_gateway_error(e):
                if attempt < max_attempts - 1:
                    wait = min(2 ** attempt, 15)
                    logger.warning(
                        "Encountered provider 400 due to blank content handling; retrying in %ss (attempt %s/%s)",
                        wait,
                        attempt + 1,
                        max_attempts,
                    )
                    time.sleep(wait)
                    continue
            logger.error(
                "LLM API status error %s (request_id=%s). Full payload written to %s",
                e.status_code,
                getattr(e, "request_id", None),
                log_path,
            )
            raise


def make_generation_node(objective: Objective, llm: ChatOpenAI, system_prompt: str):
    def generation_node(state: WorkflowState) -> WorkflowState:
        iteration = state["iteration_count"] + 1  # +1 because we increment at the end
        logger.info(f"=== Starting Iteration {iteration} ===")

        if state["iteration_count"] == 0:
            logger.info(
                "Initializing conversation with system prompt and first message"
            )
            state["messages"] = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=objective.first_message()),
            ]
        else:
            if state["validation_error"]:
                logger.warning(
                    f"Iteration {iteration}: Validation error from previous iteration: {state['validation_error']}"
                )
                state["messages"].append(
                    HumanMessage(content=state["validation_error"])
                )
                state["validation_error"] = ""

        logger.info(f"Iteration {iteration}: Calling LLM to generate molecule...")
        response = rate_limit_sensible_llm_call(llm, state["messages"])
        state["messages"].append(response)
        state["raw_model_output"] = response.content.strip()
        state["iteration_count"] += 1
        logger.info(f"Iteration {iteration}: LLM response received")

        """
        now = time.time()
        with open("timing_gpt.log", "r", encoding="utf-8") as f:
            last_line = f.readlines()[-1]

        _, last_ts = last_line.strip().split("|")
        last_time = float(last_ts)

        elapsed = now - last_time

        with open("timing_gpt.log", "a", encoding="utf-8") as f:
            f.write(f"LLM message took {elapsed:.3f}s\n")
            f.write(f"LLM message received at |{now}\n")
        """
        return state

    return generation_node


def parse_node(state: WorkflowState) -> WorkflowState:
    iteration = state["iteration_count"]
    raw = state["raw_model_output"]
    try:
        parsed = json.loads(raw)
        state["current_smiles"] = parsed["smiles"]
        state["current_reason"] = parsed["reason"]
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
    smiles = state["current_smiles"]
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


def make_prediction_node(objective: Objective):
    def prediction_node(state: WorkflowState) -> WorkflowState:
        iteration = state["iteration_count"]
        smiles = state["current_smiles"]
        logger.info(
            f"Iteration {iteration}: Calling oracle to evaluate SMILES: {smiles}"
        )

        result = objective.evaluate(state)

        score = result["score"]
        logger.info(f"Iteration {iteration}: Oracle returned score: {score:.4f}")

        # Log individual scores if this is a composite oracle
        if "scores" in result:
            score_details = ", ".join(
                [f"{k}={v:.4f}" for k, v in result["scores"].items()]
            )
            logger.info(f"Iteration {iteration}: Individual scores: {score_details}")

        state["oracle_result"] = result

        trace_entry = {
            "iteration": state["iteration_count"],
            "smiles": state["current_smiles"],
            "reason": state["current_reason"],
            "score": result["score"],
            "explanation": result["explanation"],
        }

        # If we mess with the score, save the original
        if "extra_scores" in result:
            extra_scores = json.loads(result["extra_scores"])
            if "original_affinity_probability_binary" in extra_scores:
                trace_entry["original_score"] = extra_scores[
                    "original_affinity_probability_binary"
                ]

        # Include individual scores from composite oracles
        if "scores" in result:
            trace_entry["scores"] = result["scores"]

        state["trace"].append(trace_entry)

        feedback = objective.build_feedback(state, result)
        state["messages"].append(HumanMessage(content=feedback))
        return state

    return prediction_node


def make_final_response_node(llm: ChatOpenAI, xai_mode: str | None = None):
    def final_response_node(state: WorkflowState) -> WorkflowState:
        # Extract the objective description from the first human message
        objective_context = (
            state["messages"][1].content if len(state["messages"]) > 1 else ""
        )

        # Remove explanations from trace so summary only sees what the generation
        # model actually saw (respects xai filtering in objectives)
        # For no_description mode, also remove individual scores to hide task info
        keys_to_exclude = {"explanation", "original_score"}
        if xai_mode == "no_description":
            keys_to_exclude.add("scores")

        trace_for_summary = [
            {k: v for k, v in entry.items() if k not in keys_to_exclude}
            for entry in state["trace"]
        ]

        summary_prompt = f"""
You are given the objective and trace of a molecular optimization loop.

Objective:
{objective_context}

Trace (each entry includes iteration, SMILES, reason, score):
{json.dumps(trace_for_summary, indent=2)}

Write a concise scientific summary of the optimization process.
Use ONLY the information provided. In particular, do not name the protein. Do not invent any steps or molecules.
""".strip()

        # summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary_response = rate_limit_sensible_llm_call(
            llm, [HumanMessage(content=summary_prompt)]
        )
        response_metadata = getattr(summary_response, "response_metadata", {})
        finish_reason = response_metadata.get("finish_reason", "unknown")
        logger.info(f"Final summary generated (finish_reason: {finish_reason})")
        logger.info(
            f"Content filter results: {response_metadata.get('content_filter_results')}"
        )
        logger.info(f"Token usage: {response_metadata.get('token_usage')}")
        state["final_response"] = summary_response.content
        return state

    return final_response_node
