from typing import Sequence

from .base import Oracle, OracleResult


class CompositeOracleResult(OracleResult):
    """Extended result that includes individual sub-scores."""
    scores: dict[str, float]


class CompositeOracle:
    """
    Combines multiple oracles into one.
    
    - Scores are combined as a weighted sum (normalized by sum of weights).
    - Explanations from each oracle are concatenated.
    - Individual scores are available in result["scores"] dict.
    """

    def __init__(
        self,
        oracles: Sequence[Oracle],
        weights: Sequence[float],
        names: Sequence[str] | None = None,
    ):
        if len(oracles) < 2:
            raise ValueError("CompositeOracle requires at least 2 oracles")
        
        self.oracles = list(oracles)
        self.weights = list(weights)
        self.names = list(names) if names else [f"Oracle {i+1}" for i in range(len(oracles))]
        
        if len(self.weights) != len(self.oracles):
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match "
                f"number of oracles ({len(self.oracles)})"
            )
        if len(self.names) != len(self.oracles):
            raise ValueError(
                f"Number of names ({len(self.names)}) must match "
                f"number of oracles ({len(self.oracles)})"
            )
        
        # Precompute normalized weights
        total = sum(self.weights)
        self.normalized_weights = [w / total for w in self.weights]

    def __call__(self, smiles: str) -> CompositeOracleResult:
        results = [oracle(smiles) for oracle in self.oracles]
        
        # Weighted average score
        combined_score = sum(
            w * r["score"] for w, r in zip(self.normalized_weights, results)
        )
        
        # Store individual scores by name
        scores = {name: r["score"] for name, r in zip(self.names, results)}
        
        # Concatenate explanations with simple headers
        explanation_parts = []
        for name, result in zip(self.names, results):
            explanation_parts.append(f"{name}:\n{result['explanation']}")
        
        combined_explanation = "\n\n".join(explanation_parts)
        
        return {
            "score": combined_score,
            "explanation": combined_explanation,
            "scores": scores,
        }

