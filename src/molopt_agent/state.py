from typing import TypedDict, List
from langchain_core.messages import BaseMessage

from .oracles.base import OracleResult


class WorkflowState(TypedDict):
    messages: List[BaseMessage]
    raw_model_output: str
    current_smiles: str
    current_reason: str
    oracle_result: OracleResult
    trace: list[dict] 
    iteration_count: int
    is_valid: bool
    validation_error: str
    final_response: str
    # iteration -> {"msg_index": position of its feedback message, "text": compact feedback}
    compact_feedback: dict[int, dict]
    # one entry per LLM generation call with the history policy applied
    context_log: list[dict]


def make_initial_state() -> WorkflowState:
    return {
        "messages": [],
        "raw_model_output": "",
        "current_smiles": "",
        "current_reason": "",
        "oracle_result": {},
        "trace": [],
        "iteration_count": 0,
        "is_valid": False,
        "validation_error": "",
        "final_response": "",
        "compact_feedback": {},
        "context_log": [],
    }
