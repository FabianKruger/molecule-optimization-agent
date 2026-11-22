from typing import Protocol
from ..state import WorkflowState
from ..oracles.base import OracleResult


class Objective(Protocol):
    name: str

    def first_message(self) -> str: ...
    def evaluate(self, state: WorkflowState) -> OracleResult: ...
    def build_feedback(self, state: WorkflowState, result: OracleResult) -> str: ...
    def is_done(self, state: WorkflowState, result: OracleResult) -> bool: ...
    def max_iterations(self) -> int: ...


class HistoryObjective(Objective, Protocol):
    """Objective that supports shortened explanation history (see graph/history.py).

    `history_policy` is "full", "last_k" or "best_worst_k". Objectives without
    these attributes are treated as "full". `build_compact_feedback` returns the
    feedback with the explanations replaced by a placeholder.
    """

    history_policy: str
    history_k: int

    def build_compact_feedback(self, state: WorkflowState, result: OracleResult) -> str: ...
