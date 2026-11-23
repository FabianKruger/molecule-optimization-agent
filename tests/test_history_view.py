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


def _run_sequence(policy, items, k=2):
    """items: a score per valid attempt, None for an invalid SMILES. Ends with one extra LLM call."""
    objective = SimilarityQedObjective(
        oracle=_FakeOracle(), xai=policy, history_k=k, max_iterations=100
    )
    scores = iter([i for i in items if i is not None])

    def evaluate(state):
        score = next(scores)
        return {
            "score": score,
            "explanation": "expl",
            "scores": {"Similarity": score, "QED": score},
            "explanations": {"Similarity": "SIM-EXPL", "QED": "QED-EXPL"},
        }

    objective.evaluate = evaluate
    smiles = ["C(("] * len(items) + ["C"]
    valid_count = 0
    for i, item in enumerate(items):
        if item is not None:
            valid_count += 1
            smiles[i] = "C" * valid_count
    llm = _ScriptedLLM(smiles)
    gen = make_generation_node(objective, llm, system_prompt="sys")
    pred = make_prediction_node(objective)

    state = make_initial_state()
    for _ in items:
        state = gen(state)
        state = parse_node(state)
        state = validation_node(state)
        if state["is_valid"]:
            state = pred(state)
    state = gen(state)
    return state, llm


def _roles(messages):
    return [m.type for m in messages]


def _assert_alternating(messages):
    roles = _roles(messages)
    assert roles[0] == "system"
    assert roles[1:] == ["human", "ai"] * ((len(roles) - 1) // 2) + (["human"] if len(roles) % 2 == 0 else [])


def test_truncated_last_k_removes_older_iterations_completely():
    state, llm = _run_sequence("truncated_last_k", [0.1, 0.2, 0.3, 0.4, 0.5], k=2)
    prompt = llm.calls[-1]
    _assert_alternating(prompt)
    assert prompt[-1].type == "human"
    assert len(prompt) == 2 + 2 * 2  # system + task + two assistant/feedback pairs
    text = " ".join(m.content for m in prompt)
    assert "Iteration: 4 / 100" in text and "Iteration: 5 / 100" in text
    assert "Iteration: 3 / 100" not in text and "Iteration: 1 / 100" not in text
    assert len(_explained_in(prompt)) == 2
    assert all("Explanation omitted" not in m.content for m in prompt)
    entry = state["context_log"][-1]
    assert entry["form"] == "truncated"
    assert entry["explained_iterations"] == [4, 5]
    assert entry["dropped_iterations"] == [1, 2, 3]
    assert entry["counter_appended"] is False
    assert len(state["messages"]) == 2 + 2 * 5 + 1  # full conversation untouched


def test_truncated_equals_full_before_budget_is_reached():
    state, llm = _run_sequence("truncated_last_k", [0.1, 0.2], k=3)
    assert [m.content for m in llm.calls[-1]] == [m.content for m in state["messages"][:-1]]
    state, llm = _run_sequence("truncated_best_worst_k", [0.1, 0.2, 0.3, 0.4], k=2)
    assert [m.content for m in llm.calls[-1]] == [m.content for m in state["messages"][:-1]]


def test_truncated_best_worst_keeps_only_best_and_worst_and_adds_counter():
    state, llm = _run_sequence("truncated_best_worst_k", [0.9, 0.1, 0.8, 0.2, 0.4, 0.5], k=1)
    prompt = llm.calls[-1]
    _assert_alternating(prompt)
    entry = state["context_log"][-1]
    assert entry["explained_iterations"] == [1, 2]
    assert entry["dropped_iterations"] == [3, 4, 5, 6]
    assert entry["best_iterations"] == [1] and entry["worst_iterations"] == [2]
    assert entry["counter_appended"] is True
    assert prompt[-1].content.endswith("Iteration: 6 / 100")
    # the counter is only in the copy sent to the LLM
    stored_feedback = [m for m in state["messages"] if m.type == "human"][-1].content
    assert stored_feedback.count("Iteration: 6 / 100") == 1 and not stored_feedback.endswith("Iteration: 6 / 100")
    assert sum("Iteration: " in m.content for m in prompt if m.type == "human") == 2  # feedback 2 + counter line
    assert len(_explained_in(prompt)) == 2


def test_truncated_best_worst_shows_latest_when_it_is_best():
    state, llm = _run_sequence("truncated_best_worst_k", [0.5, 0.5, 0.5, 0.5, 0.9], k=1)
    entry = state["context_log"][-1]
    assert 5 in entry["explained_iterations"]
    assert entry["counter_appended"] is False


def test_truncated_drops_old_invalid_attempts_but_keeps_the_latest():
    # call 2 is invalid (old), call 5 is invalid and the most recent attempt
    state, llm = _run_sequence("truncated_last_k", [0.1, None, 0.2, 0.3, None], k=2)
    prompt = llm.calls[-1]
    entry = state["context_log"][-1]
    assert entry["explained_iterations"] == [3, 4]
    assert entry["dropped_iterations"] == [1, 2]
    _assert_alternating(prompt)
    assert prompt[-1].content.startswith("Invalid SMILES")
    assert len(prompt) == 2 + 2 * 3  # two valid pairs + the latest invalid attempt with its error


def test_truncated_older_invalid_attempt_removed_even_if_its_retry_is_kept():
    state, llm = _run_sequence("truncated_last_k", [None, 0.1, 0.2, 0.3], k=3)
    entry = state["context_log"][-1]
    assert entry["explained_iterations"] == [2, 3, 4]
    assert entry["dropped_iterations"] == [1]
    assert all(not m.content.startswith("Invalid SMILES") for m in llm.calls[-1])


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
    test_truncated_last_k_removes_older_iterations_completely()
    test_truncated_equals_full_before_budget_is_reached()
    test_truncated_best_worst_keeps_only_best_and_worst_and_adds_counter()
    test_truncated_best_worst_shows_latest_when_it_is_best()
    test_truncated_drops_old_invalid_attempts_but_keeps_the_latest()
    test_truncated_older_invalid_attempt_removed_even_if_its_retry_is_kept()
    print("ok")
