"""Tests for oracle/objective registry lookup (no heavy oracle execution)."""

from __future__ import annotations

import pytest

rdkit = pytest.importorskip("rdkit")  # noqa: F401

from molopt_agent.config import ObjectiveConfig, OracleConfig  # noqa: E402
from molopt_agent.objectives import (  # noqa: E402
    OBJECTIVE_REGISTRY,
    build_objective_from_config,
)
from molopt_agent.oracles import ORACLE_REGISTRY, build_oracle_from_config  # noqa: E402
from molopt_agent.oracles.composite import CompositeOracle  # noqa: E402
from molopt_agent.state import make_initial_state  # noqa: E402


def test_core_oracles_registered():
    for name in ("qed", "similarity", "composite"):
        assert name in ORACLE_REGISTRY


def test_core_objectives_registered():
    for name in ("qed", "similarity", "similarity_qed"):
        assert name in OBJECTIVE_REGISTRY


def test_build_qed_oracle_and_objective():
    oracle = build_oracle_from_config(OracleConfig(name="qed", params={}))
    objective = build_objective_from_config(
        ObjectiveConfig(name="qed", params={"target_qed": 0.5, "max_iterations": 3}),
        oracle=oracle,
    )
    assert objective.name == "qed"
    assert objective.max_iterations() == 3

    state = make_initial_state()
    state["current_smiles"] = "CCO"
    result = objective.evaluate(state)
    assert 0.0 <= result["score"] <= 1.0
    assert isinstance(result["explanation"], str)
    assert "QED" in objective.first_message()


def test_build_similarity_oracle():
    oracle = build_oracle_from_config(
        OracleConfig(name="similarity", params={"target_smiles": "CCO"})
    )
    result = oracle("CCO")
    assert result["score"] == 0.0
    assert "IDENTICAL" in result["explanation"]

    result_other = oracle("c1ccccc1")
    assert 0.0 <= result_other["score"] <= 1.0


def test_unknown_oracle_raises():
    with pytest.raises(ValueError, match="Unknown oracle"):
        build_oracle_from_config(OracleConfig(name="not-an-oracle", params={}))


def test_unknown_objective_raises():
    oracle = build_oracle_from_config(OracleConfig(name="qed", params={}))
    with pytest.raises(ValueError, match="Unknown objective"):
        build_objective_from_config(
            ObjectiveConfig(name="not-an-objective", params={}),
            oracle=oracle,
        )


def test_composite_oracle_requires_nested_config():
    with pytest.raises(ValueError, match="oracles"):
        build_oracle_from_config(OracleConfig(name="composite", params={}))

    with pytest.raises(ValueError, match="weights"):
        build_oracle_from_config(
            OracleConfig(
                name="composite",
                params={"oracles": [{"name": "qed", "params": {}}]},
            )
        )


def test_build_composite_qed_similarity():
    oracle = build_oracle_from_config(
        OracleConfig(
            name="composite",
            params={
                "oracles": [
                    {"name": "qed", "params": {}},
                    {"name": "similarity", "params": {"target_smiles": "CCO"}},
                ],
                "weights": [0.5, 0.5],
                "names": ["qed", "sim"],
            },
        )
    )
    assert isinstance(oracle, CompositeOracle)
    result = oracle("CCN")
    assert "scores" in result
    assert set(result["scores"]) == {"qed", "sim"}
    assert 0.0 <= result["score"] <= 1.0


def test_qed_objective_feedback_and_done():
    oracle = build_oracle_from_config(OracleConfig(name="qed", params={}))
    objective = build_objective_from_config(
        ObjectiveConfig(name="qed", params={"target_qed": 0.01, "max_iterations": 2}),
        oracle=oracle,
    )
    state = make_initial_state()
    state["current_smiles"] = "CCO"
    state["iteration_count"] = 1
    result = objective.evaluate(state)
    feedback = objective.build_feedback(state, result)
    assert "QED:" in feedback
    assert objective.is_done(state, result) is True  # low target

    objective_hard = build_objective_from_config(
        ObjectiveConfig(name="qed", params={"target_qed": 0.999, "max_iterations": 2}),
        oracle=oracle,
    )
    assert objective_hard.is_done(state, result) is False
    state["iteration_count"] = 2
    assert objective_hard.is_done(state, result) is True


def test_composite_oracle_direct_validation_errors():
    qed = build_oracle_from_config(OracleConfig(name="qed", params={}))
    with pytest.raises(ValueError, match="at least 2"):
        CompositeOracle(oracles=[qed], weights=[1.0])
    with pytest.raises(ValueError, match="weights"):
        CompositeOracle(oracles=[qed, qed], weights=[1.0])
    with pytest.raises(ValueError, match="names"):
        CompositeOracle(oracles=[qed, qed], weights=[0.5, 0.5], names=["only_one"])


def test_similarity_oracle_invalid_target():
    with pytest.raises(ValueError, match="Invalid target SMILES"):
        build_oracle_from_config(
            OracleConfig(name="similarity", params={"target_smiles": "not_a_smiles"})
        )


def test_oracle_bad_constructor_params():
    with pytest.raises(ValueError, match="Failed to construct oracle"):
        build_oracle_from_config(
            OracleConfig(name="similarity", params={"unexpected": True})
        )
