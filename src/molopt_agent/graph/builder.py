import os
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from ..state import WorkflowState
from ..config import ExperimentConfig
from ..objectives.base import Objective
from ..system_prompt import SYSTEM_PROMPT
from .nodes import (
    make_generation_node,
    parse_node,
    validation_node,
    make_prediction_node,
    make_final_response_node,
)
from .routing import (
    make_route_after_parse,
    make_route_after_validation,
    make_route_after_prediction,
)


API_KEY_ENV = "OPENAI_API_KEY"


def build_workflow(objective: Objective, cfg: ExperimentConfig):
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Missing API key: set {API_KEY_ENV} environment variable"
        )

    llm_kwargs = {
        "model": cfg.llm.model,
        "api_key": api_key,
    }
    if cfg.llm.base_url:
        llm_kwargs["base_url"] = cfg.llm.base_url

    llm_gen = ChatOpenAI(temperature=cfg.llm.temperature, **llm_kwargs)
    llm_sum = ChatOpenAI(temperature=0.0, **llm_kwargs)

    gen_node = make_generation_node(objective, llm_gen, system_prompt=SYSTEM_PROMPT)
    pred_node = make_prediction_node(objective)
    final_node = make_final_response_node(llm_sum)

    route_parse = make_route_after_parse(objective)
    route_val = make_route_after_validation(objective)
    route_pred = make_route_after_prediction(objective)

    g = StateGraph(WorkflowState)

    g.add_node("generation", gen_node)
    g.add_node("parse", parse_node)
    g.add_node("validation", validation_node)
    g.add_node("prediction", pred_node)
    g.add_node("final", final_node)

    g.set_entry_point("generation")

    g.add_edge("generation", "parse")
    g.add_conditional_edges(
        "parse",
        route_parse,
        {"validation": "validation", "generation": "generation", "final": "final"},
    )
    g.add_conditional_edges(
        "validation",
        route_val,
        {"prediction": "prediction", "generation": "generation", "final": "final"},
    )
    g.add_conditional_edges(
        "prediction",
        route_pred,
        {"generation": "generation", "final": "final"},
    )
    g.add_edge("final", END)

    return g.compile()
