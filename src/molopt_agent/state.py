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
    terminated_early: bool
    termination_reason: str
    generation_error: str
    last_llm_start_ts: float
    last_llm_wall_time_s: float
    last_prompt_tokens: int
    last_completion_tokens: int
    last_total_tokens: int


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
        "terminated_early": False,
        "termination_reason": "",
        "generation_error": "",
        "last_llm_start_ts": 0.0,
        "last_llm_wall_time_s": 0.0,
        "last_prompt_tokens": 0,
        "last_completion_tokens": 0,
        "last_total_tokens": 0,
    }
