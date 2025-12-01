from typing import Literal

from ..state import WorkflowState
from ..objectives.base import Objective


def make_route_after_parse(objective: Objective):
    def route_after_parse(state: WorkflowState) -> Literal["validation", "generation", "final"]:
        if state["iteration_count"] >= objective.max_iterations():
            return "final"
        if state["current_smiles"] == "" or not state["is_valid"]:
            return "generation"
        return "validation"
    return route_after_parse


def make_route_after_validation(objective: Objective):
    def route_after_validation(state: WorkflowState) -> Literal["prediction", "generation", "final"]:
        if state["iteration_count"] >= objective.max_iterations():
            return "final"
        if state["is_valid"]:
            return "prediction"
        return "generation"
    return route_after_validation


def make_route_after_prediction(objective: Objective):
    def route_after_prediction(state: WorkflowState) -> Literal["generation", "final"]:
        result = state.get("oracle_result", {})
        if result and objective.is_done(state, result):
            return "final"
        if state["iteration_count"] >= objective.max_iterations():
            return "final"
        return "generation"
    return route_after_prediction
