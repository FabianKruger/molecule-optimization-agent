from typing import Sequence, TypedDict

from .base import Oracle, OracleResult


class CompositeComponent(TypedDict):
    """Select one score from a child oracle and assign a Combined weight."""

    oracle: int  # index into ``oracles``
    score: str  # key in that oracle's OracleResult
    weight: float


class CompositeOracle:
    """
    Combines scores from multiple oracles.

    Each component picks one score key from one child oracle. Those scores are
    exposed in the result under the same key. ``Combined`` is the normalized
    weighted average (weight 0 scores are included for constraints but do not
    affect ``Combined`` when all positive weights are used for normalization).
    """

    def __init__(
        self,
        oracles: Sequence[Oracle],
        components: Sequence[CompositeComponent],
    ):
        if len(oracles) < 1:
            raise ValueError("CompositeOracle requires at least 1 oracle")
        if len(components) < 1:
            raise ValueError("CompositeOracle requires at least 1 component")

        self.oracles = list(oracles)
        self.components = [dict(c) for c in components]

        seen_keys: set[str] = set()
        for i, comp in enumerate(self.components):
            if "oracle" not in comp or "score" not in comp or "weight" not in comp:
                raise ValueError(
                    f"Component {i} must have keys 'oracle', 'score', and 'weight'"
                )
            idx = int(comp["oracle"])
            if idx < 0 or idx >= len(self.oracles):
                raise ValueError(
                    f"Component {i} oracle index {idx} out of range "
                    f"[0, {len(self.oracles)})"
                )
            key = str(comp["score"])
            if key in seen_keys:
                raise ValueError(f"Duplicate component score key '{key}'")
            if key == "Combined":
                raise ValueError("Component score key 'Combined' is reserved")
            seen_keys.add(key)
            comp["oracle"] = idx
            comp["score"] = key
            comp["weight"] = float(comp["weight"])

        total = sum(c["weight"] for c in self.components)
        if total > 0:
            self.normalized_weights = [c["weight"] / total for c in self.components]
        else:
            self.normalized_weights = [0.0] * len(self.components)

    def get_params(self) -> dict:
        """Return oracle configuration for sharing with objectives."""
        return {
            "components": self.components,
            "normalized_weights": self.normalized_weights,
            "oracles": self.oracles,
        }

    def __call__(self, smiles: str) -> OracleResult:
        results = [oracle(smiles) for oracle in self.oracles]

        scores: OracleResult = {}
        for comp in self.components:
            idx = comp["oracle"]
            key = comp["score"]
            child = results[idx]
            if key not in child:
                raise KeyError(
                    f"Oracle {idx} result missing score '{key}'. "
                    f"Available: {list(child)}"
                )
            scores[key] = child[key]

        combined = sum(
            w * scores[c["score"]]["value"]
            for c, w in zip(self.components, self.normalized_weights)
        )
        scores["Combined"] = {"value": combined}

        return scores
