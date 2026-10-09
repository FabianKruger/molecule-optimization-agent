"""Unit tests for oracle result shape and CompositeOracle."""

from __future__ import annotations

import pytest

from seismo.oracles.composite import CompositeOracle
from seismo.oracles.qed import ExplainableQedOracle


def _const(key: str, value: float):
    """Return a stub oracle that always yields one named score."""

    def oracle(smiles: str):
        return {key: {"value": value, "explanation": f"{key}={value}"}}

    return oracle


class TestCompositeOracle:
    def test_weighted_combined(self):
        oracle = CompositeOracle(
            oracles=[_const("A", 1.0), _const("B", 0.0)],
            components=[
                {"oracle": 0, "score": "A", "weight": 1.0},
                {"oracle": 1, "score": "B", "weight": 3.0},
            ],
        )
        result = oracle("C")

        assert result["A"]["value"] == 1.0
        assert result["B"]["value"] == 0.0
        assert result["Combined"]["value"] == pytest.approx(0.25)
        assert "explanation" in result["A"]

    def test_zero_weight_included_but_ignored_in_combined(self):
        oracle = CompositeOracle(
            oracles=[_const("A", 1.0), _const("Constraint", 0.0)],
            components=[
                {"oracle": 0, "score": "A", "weight": 1.0},
                {"oracle": 1, "score": "Constraint", "weight": 0.0},
            ],
        )
        result = oracle("C")

        assert result["Constraint"]["value"] == 0.0
        assert result["Combined"]["value"] == pytest.approx(1.0)

    def test_missing_child_score_raises(self):
        oracle = CompositeOracle(
            oracles=[_const("A", 1.0)],
            components=[{"oracle": 0, "score": "Missing", "weight": 1.0}],
        )
        with pytest.raises(KeyError, match="Missing"):
            oracle("C")

    def test_rejects_duplicate_and_reserved_keys(self):
        with pytest.raises(ValueError, match="Duplicate"):
            CompositeOracle(
                oracles=[_const("A", 1.0), _const("A", 0.5)],
                components=[
                    {"oracle": 0, "score": "A", "weight": 1.0},
                    {"oracle": 1, "score": "A", "weight": 1.0},
                ],
            )
        with pytest.raises(ValueError, match="reserved"):
            CompositeOracle(
                oracles=[_const("A", 1.0)],
                components=[{"oracle": 0, "score": "Combined", "weight": 1.0}],
            )


class TestQedOracleShape:
    def test_result_shape(self):
        result = ExplainableQedOracle()("CCO")  # ethanol

        assert set(result) == {"QED"}
        assert isinstance(result["QED"]["value"], float)
        assert 0.0 <= result["QED"]["value"] <= 1.0
        assert isinstance(result["QED"]["explanation"], str)
        assert result["QED"]["explanation"]
