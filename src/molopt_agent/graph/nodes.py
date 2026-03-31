import json
import logging
import random
import time

import httpcore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, BadRequestError, RateLimitError
from rdkit import Chem

from ..objectives.base import Objective
from ..state import WorkflowState

logger = logging.getLogger(__name__)

# Stable error codes used by native OpenAI for context overflow.
_CONTEXT_WINDOW_CODES = {"context_length_exceeded", "context_window_exceeded"}
# Phrase present in this deployment's proxy error body / message.
_CONTEXT_WINDOW_PHRASES = ("context window exceeded",)


def _extract_error_message(error: Exception) -> str:
    """Return the most descriptive text available from a provider error."""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        # Proxy pattern:  {"detail": "Context window exceeded ..."}
        detail = body.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        # Native OpenAI:  {"error": {"message": "...", "code": "..."}}
        nested = body.get("error")
        if isinstance(nested, dict):
            message = nested.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()

    message = getattr(error, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()

    text = str(error).strip()
    if text:
        return text

    return error.__class__.__name__


def is_context_window_error(error: Exception) -> bool:
    if not isinstance(error, BadRequestError):
        return False
    # Primary: stable machine-readable code (native OpenAI endpoints).
    if getattr(error, "code", None) in _CONTEXT_WINDOW_CODES:
        return True
    # Secondary: body detail or message text (proxy / Anthropic-compatible endpoints).
    text = _extract_error_message(error).lower()
    return any(phrase in text for phrase in _CONTEXT_WINDOW_PHRASES)


def _mark_context_window_stop(state: WorkflowState, error: Exception) -> None:
    error_message = _extract_error_message(error)
    state["terminated_early"] = True
    state["termination_reason"] = "context_window_exceeded"
    state["generation_error"] = error_message
    state["raw_model_output"] = ""
    state["current_smiles"] = ""
    state["current_reason"] = ""
    state["is_valid"] = False
    state["validation_error"] = ""


def rate_limit_sensible_llm_call(llm: ChatOpenAI, message, max_attempts=3):
    delay = 60  # sensible default for hard limits
    conn_delay = 0.5

    for attempt in range(max_attempts):
        try:
            return llm.invoke(message)

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


def make_generation_node(objective: Objective, llm: ChatOpenAI, system_prompt: str):
    def generation_node(state: WorkflowState) -> WorkflowState:
        iteration = state["iteration_count"] + 1  # +1 because we increment at the end
        logger.info(f"=== Starting Iteration {iteration} ===")
        state["terminated_early"] = False
        state["termination_reason"] = ""
        state["generation_error"] = ""

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
        _llm_t0 = time.time()
        state["last_llm_start_ts"] = _llm_t0
        try:
            response = rate_limit_sensible_llm_call(llm, state["messages"])
        except Exception as error:
            # Always record wall time regardless of error type.
            state["last_llm_wall_time_s"] = round(time.time() - _llm_t0, 3)
            if is_context_window_error(error):
                _mark_context_window_stop(state, error)
                logger.warning(
                    "Iteration %s: Context window exceeded, routing to final summary: %s",
                    iteration,
                    state["generation_error"],
                )
            else:
                # Any other LLM error (timeout, auth, unexpected HTTP, etc.): still
                # mark terminated_early so the graph routes to the final node and
                # the conversation+trace are saved.  Never re-raise from here.
                error_message = _extract_error_message(error)
                state["terminated_early"] = True
                state["termination_reason"] = "generation_error"
                state["generation_error"] = f"{type(error).__name__}: {error_message}"
                state["raw_model_output"] = ""
                state["current_smiles"] = ""
                state["current_reason"] = ""
                state["is_valid"] = False
                state["validation_error"] = ""
                logger.error(
                    "Iteration %s: Unexpected LLM error, routing to final summary: %s",
                    iteration,
                    error_message,
                    exc_info=True,
                )
            return state

        state["last_llm_wall_time_s"] = round(time.time() - _llm_t0, 3)
        token_usage = (getattr(response, "response_metadata", {}) or {}).get("token_usage", {}) or {}
        state["last_prompt_tokens"] = token_usage.get("prompt_tokens", 0)
        state["last_completion_tokens"] = token_usage.get("completion_tokens", 0)
        state["last_total_tokens"] = token_usage.get("total_tokens", 0)
        state["messages"].append(response)
        state["raw_model_output"] = response.content.strip()
        state["iteration_count"] += 1
        logger.info(
            "Iteration %s: LLM response received (prompt_tokens=%s, completion_tokens=%s, wall_time=%.2fs)",
            iteration,
            state["last_prompt_tokens"],
            state["last_completion_tokens"],
            state["last_llm_wall_time_s"],
        )
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

        _oracle_t0 = time.time()
        result = objective.evaluate(state)
        _oracle_wall_s = round(time.time() - _oracle_t0, 3)

        score = result["score"]
        logger.info(
            "Iteration %s: Oracle returned score: %.4f (wall_time=%.2fs)",
            iteration,
            score,
            _oracle_wall_s,
        )

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
            "llm_start_ts": round(state.get("last_llm_start_ts", 0.0), 3),
            "llm_wall_time_s": state.get("last_llm_wall_time_s", 0.0),
            "prompt_tokens": state.get("last_prompt_tokens", 0),
            "completion_tokens": state.get("last_completion_tokens", 0),
            "total_tokens": state.get("last_total_tokens", 0),
            "oracle_start_ts": round(_oracle_t0, 3),
            "oracle_wall_time_s": _oracle_wall_s,
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

        stop_note = ""
        termination_reason = state.get("termination_reason", "")
        if termination_reason == "context_window_exceeded":
            stop_note = f"""

The optimization loop stopped early because the generation prompt exceeded the model context window.
Provider error: {state.get("generation_error", "")}"""
        elif termination_reason == "generation_error":
            stop_note = f"""

The optimization loop stopped early due to an unexpected error during LLM generation.
Error: {state.get("generation_error", "")}"""

        summary_prompt = f"""
You are given the objective and trace of a molecular optimization loop.

Objective:
{objective_context}
{stop_note}

Trace (each entry includes iteration, SMILES, reason, score):
{json.dumps(trace_for_summary, indent=2)}

Write a concise scientific summary of the optimization process.
Use ONLY the information provided. In particular, do not name the protein. Do not invent any steps or molecules.
""".strip()

        try:
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
        except Exception as error:
            logger.warning(
                "Falling back to a local final summary after summary generation failed: %s",
                _extract_error_message(error),
            )
            _reason = state.get("termination_reason", "")
            if _reason == "context_window_exceeded":
                state["final_response"] = (
                    "Optimization stopped because the model context window was reached. "
                    f"Completed {state['iteration_count']} iterations before termination. "
                    "The conversation and trace were preserved."
                )
            elif _reason == "generation_error":
                state["final_response"] = (
                    f"Optimization stopped after {state['iteration_count']} iterations due to an "
                    f"unexpected LLM error ({state.get('generation_error', 'unknown')}). "
                    "The conversation and trace were preserved."
                )
            else:
                state["final_response"] = (
                    "Optimization finished, but the final summary call failed. "
                    f"Completed {state['iteration_count']} iterations. "
                    f"Summary error: {_extract_error_message(error)}"
                )
        return state

    return final_response_node
