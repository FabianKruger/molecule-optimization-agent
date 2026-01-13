from typing import Literal
import logging

from ..state import WorkflowState
from ..objectives.base import Objective

logger = logging.getLogger(__name__)


def make_route_after_parse(objective: Objective):
    def route_after_parse(state: WorkflowState) -> Literal["validation", "generation", "final"]:
        iteration = state["iteration_count"]
        if state["iteration_count"] >= objective.max_iterations():
            logger.info(f"Iteration {iteration}: Reached max iterations, moving to final summary")
            return "final"
        if state["current_smiles"] == "" or not state["is_valid"]:
            logger.info(f"Iteration {iteration}: Parse failed, retrying generation")
            return "generation"
        logger.debug(f"Iteration {iteration}: Parse successful, moving to validation")
        return "validation"
    return route_after_parse


def make_route_after_validation(objective: Objective):
    def route_after_validation(state: WorkflowState) -> Literal["prediction", "generation", "final"]:
        iteration = state["iteration_count"]
        if state["iteration_count"] >= objective.max_iterations():
            logger.info(f"Iteration {iteration}: Reached max iterations, moving to final summary")
            return "final"
        if state["is_valid"]:
            logger.debug(f"Iteration {iteration}: Validation successful, moving to prediction")
            return "prediction"
        logger.info(f"Iteration {iteration}: Validation failed, retrying generation")
        return "generation"
    return route_after_validation


def make_route_after_prediction(objective: Objective):
    def route_after_prediction(state: WorkflowState) -> Literal["generation", "final"]:
        iteration = state["iteration_count"]
        result = state.get("oracle_result", {})
        if result and objective.is_done(state, result):
            logger.info(f"Iteration {iteration}: Objective achieved! Moving to final summary")
            return "final"
        if state["iteration_count"] >= objective.max_iterations():
            logger.info(f"Iteration {iteration}: Reached max iterations, moving to final summary")
            return "final"
        logger.debug(f"Iteration {iteration}: Continuing optimization")
        return "generation"
    return route_after_prediction
