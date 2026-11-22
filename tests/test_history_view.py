"""Tests for the shortened-history modes (last_k, best_worst_k).

Run directly with `pixi run python tests/test_history_view.py`.
"""

import json

from langchain_core.messages import AIMessage

from molopt_agent.graph.history import select_explained
from molopt_agent.graph.nodes import (
    make_generation_node,
    make_prediction_node,
    parse_node,
    validation_node,
)
from molopt_agent.objectives.similarity_qed import SimilarityQedObjective
from molopt_agent.state import make_initial_state


def _trace(scores):
    return [{"iteration": i + 1, "score": s} for i, s in enumerate(scores)]


def test_last_k_keeps_most_recent_k():
    sel = select_explained(_trace([0.1] * 5), "last_k", 2)
    assert sel["explained"] == [4, 5]
    assert sel["omitted"] == [1, 2, 3]


def test_last_k_equals_full_before_k_iterations():
    sel = select_explained(_trace([0.1, 0.2]), "last_k", 2)
    assert sel["omitted"] == []


def test_best_worst_k_selects_by_score_without_latest_bias():
    # Latest (iteration 6, score 0.5) is in the middle, so it is not explained.
    sel = select_explained(_trace([0.9, 0.1, 0.8, 0.2, 0.4, 0.5]), "best_worst_k", 2)
    assert sel["best"] == [1, 3]
    assert sel["worst"] == [2, 4]
    assert sel["explained"] == [1, 2, 3, 4]
    assert sel["omitted"] == [5, 6]


def test_best_worst_k_latest_explained_when_extreme():
    sel = select_explained(_trace([0.5, 0.5, 0.5, 0.5, 0.9]), "best_worst_k", 1)
    assert 5 in sel["explained"]


def test_best_worst_k_equals_full_up_to_2k_iterations():
    sel = select_explained(_trace([0.1, 0.2, 0.3, 0.4]), "best_worst_k", 2)
    assert sel["omitted"] == []
    assert sel["best"] == [4, 3]
    assert sel["worst"] == [1, 2]


def test_best_worst_k_ties_prefer_earlier_iteration():
    sel = select_explained(_trace([0.5] * 6), "best_worst_k", 1)
    assert sel["best"] == [1]
    assert sel["worst"] == [1]
    assert sel["explained"] == [1]


class _FakeOracle:
    def get_params(self):
        class Sub:
            def get_params(self):
                return {"target_smiles": "CCO"}

        return {"weights": [0.5, 0.5], "oracles": [Sub()]}

    def __call__(self, smiles):
        raise NotImplementedError


class _ScriptedLLM:
    """Returns the scripted SMILES one by one and records what it was sent."""

    def __init__(self, smiles):
        self._smiles = iter(smiles)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content=json.dumps({"reason": "r", "smiles": next(self._smiles)}))


def _run(policy, scores, k=2):
    objective = SimilarityQedObjective(
        oracle=_FakeOracle(), xai=policy, history_k=k, max_iterations=len(scores)
    )
    results = iter(scores)

    def evaluate(state):
        score = next(results)
        return {
            "score": score,
            "explanation": "expl",
            "scores": {"Similarity": score, "QED": score},
            "explanations": {"Similarity": "SIM-EXPL", "QED": "QED-EXPL"},
        }

    objective.evaluate = evaluate
    llm = _ScriptedLLM(["C" * (i + 1) for i in range(len(scores) + 1)])
    gen = make_generation_node(objective, llm, system_prompt="sys")
    pred = make_prediction_node(objective)

    state = make_initial_state()
    for _ in scores:
        state = gen(state)
        state = parse_node(state)
        state = validation_node(state)
        state = pred(state)
    state = gen(state)  # one more call so the last feedback is part of a prompt
    return state, llm


def _explained_in(messages):
    return [m for m in messages if "SIM-EXPL" in m.content]


def test_last_k_prompt_and_log():
    state, llm = _run("last_k", [0.1, 0.2, 0.3, 0.4, 0.5], k=2)
    last_prompt = llm.calls[-1]
    assert len(_explained_in(last_prompt)) == 2
    assert sum("Explanation omitted" in m.content for m in last_prompt) == 3
    entry = state["context_log"][-1]
    assert entry["explained_iterations"] == [4, 5]
    assert entry["omitted_iterations"] == [1, 2, 3]
    assert entry["generation_call"] == 6
    # Full conversation is untouched
    assert len(_explained_in(state["messages"])) == 5
    assert all(set(e) <= {"iteration", "smiles", "reason", "score", "explanation", "scores"} for e in state["trace"])
    assert all("Explanation omitted" not in m.content for m in state["messages"])


def test_best_worst_k_prompt_and_log():
    state, llm = _run("best_worst_k", [0.9, 0.1, 0.8, 0.2, 0.4, 0.5], k=2)
    entry = state["context_log"][-1]
    assert entry["best_iterations"] == [1, 3]
    assert entry["worst_iterations"] == [2, 4]
    assert len(_explained_in(llm.calls[-1])) == 4
    placeholder = [m for m in llm.calls[-1] if "Explanation omitted" in m.content]
    assert len(placeholder) == 2
    assert "best or 2 worst" in placeholder[0].content


def test_full_policy_logs_nothing_and_sends_everything():
    state, llm = _run("full", [0.1, 0.2, 0.3], k=1)
    assert state["context_log"] == []
    assert len(_explained_in(llm.calls[-1])) == 3


def test_invalid_history_k():
    try:
        SimilarityQedObjective(oracle=_FakeOracle(), xai="last_k", history_k=0)
    except ValueError:
        return
    raise AssertionError("history_k=0 should be rejected")


if __name__ == "__main__":
    test_last_k_keeps_most_recent_k()
    test_last_k_equals_full_before_k_iterations()
    test_best_worst_k_selects_by_score_without_latest_bias()
    test_best_worst_k_latest_explained_when_extreme()
    test_best_worst_k_equals_full_up_to_2k_iterations()
    test_best_worst_k_ties_prefer_earlier_iteration()
    test_last_k_prompt_and_log()
    test_best_worst_k_prompt_and_log()
    test_full_policy_logs_nothing_and_sends_everything()
    test_invalid_history_k()
    print("ok")
