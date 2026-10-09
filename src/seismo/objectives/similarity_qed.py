from typing import Literal

from ..oracles.base import Oracle, OracleResult
from ..state import WorkflowState
from .generic_messages import get_generic_feedback, get_generic_first_message

FIRST_MESSAGE_TEMPLATE = """
We are optimizing molecules for TWO objectives simultaneously:

1. Structural Similarity to a target molecule using MACCS fingerprints
   - Target molecule SMILES: {target_smiles}
   - Minimum required: similarity ≥ {min_similarity:.2f}
   - IMPORTANT: You must NOT propose the exact target molecule.

2. Drug-likeness using the QED score
   - Minimum required: QED ≥ {min_qed:.2f}
   - QED evaluates: molecular weight, LogP, H-bond donors/acceptors, polar surface area, rotatable bonds, aromatic rings, structural alerts.

The combined score is: {similarity_weight} * similarity + {qed_weight} * QED
Target combined score: ≥ {target_score:.2f}

Success requires:
- Combined score ≥ {target_score:.2f}
- Similarity ≥ {min_similarity:.2f}
- QED ≥ {min_qed:.2f}

Step 1:
Propose a single initial molecule as a SMILES string that balances both objectives.

Respond with a single JSON object:
{{
  "reason": "<why this molecule should satisfy both objectives>",
  "smiles": "<SMILES string - must be different from target>"
}}
""".strip()


class SimilarityQedObjective:
    """Multi-objective optimization for Tanimoto similarity and QED."""

    name = "similarity_qed"

    # Keys used in the CompositeOracle's scores dict
    SIMILARITY_KEY = "Similarity"
    QED_KEY = "QED"

    def __init__(
        self,
        oracle: Oracle,
        target_score: float = 0.75,
        min_similarity: float = 0.6,
        min_qed: float = 0.5,
        max_iterations: int = 20,
        xai: Literal["full", "partial", "none", "no_description"] = "full",
    ):
        self.oracle = oracle
        self._target_score = float(target_score)
        self._min_similarity = float(min_similarity)
        self._min_qed = float(min_qed)
        self._max_iterations = int(max_iterations)
        self._xai = xai

        # Extract from oracle
        oracle_params = self._extract_from_oracle(oracle)
        self._target_smiles = oracle_params["target_smiles"]
        self._similarity_weight = float(oracle_params["similarity_weight"])
        self._qed_weight = float(oracle_params["qed_weight"])

    def _extract_from_oracle(self, oracle: Oracle) -> dict:
        """Extract shared params from oracle via get_params()."""
        if not hasattr(oracle, "get_params"):
            raise ValueError(
                "Oracle must implement get_params() for SimilarityQedObjective"
            )

        params = oracle.get_params()
        extracted = {}

        components = params.get("components")
        if not components:
            raise ValueError("Oracle must provide 'components' in get_params()")
        weights_by_score = {c["score"]: float(c["weight"]) for c in components}
        if (
            self.SIMILARITY_KEY not in weights_by_score
            or self.QED_KEY not in weights_by_score
        ):
            raise ValueError(
                f"Composite components must include '{self.SIMILARITY_KEY}' and '{self.QED_KEY}'"
            )
        extracted["similarity_weight"] = weights_by_score[self.SIMILARITY_KEY]
        extracted["qed_weight"] = weights_by_score[self.QED_KEY]

        # Check sub-oracles for target_smiles
        for sub_oracle in params.get("oracles", []):
            if hasattr(sub_oracle, "get_params"):
                sub_params = sub_oracle.get_params()
                if "target_smiles" in sub_params:
                    extracted["target_smiles"] = sub_params["target_smiles"]
                    break

        if "target_smiles" not in extracted:
            raise ValueError(
                "Could not extract target_smiles from oracle's sub-oracles"
            )

        return extracted

    def first_message(self) -> str:
        if self._xai == "no_description":
            return get_generic_first_message(higher_is_better=True)
        return FIRST_MESSAGE_TEMPLATE.format(
            target_smiles=self._target_smiles,
            target_score=self._target_score,
            min_similarity=self._min_similarity,
            min_qed=self._min_qed,
            similarity_weight=self._similarity_weight,
            qed_weight=self._qed_weight,
        )

    def evaluate(self, state: WorkflowState) -> OracleResult:
        smiles = state["current_smiles"]
        return self.oracle(smiles)

    def _get_explanation(self, result: OracleResult) -> str:
        """Get explanation based on xai setting."""
        similarity_explanation = result.get(self.SIMILARITY_KEY, {}).get(
            "explanation", ""
        )
        qed_explanation = result.get(self.QED_KEY, {}).get("explanation", "")

        if self._xai == "none":
            similarity_text = "No explanation provided."
            qed_text = "No explanation provided."
        elif self._xai == "partial":
            similarity_text = "No explanation provided."
            qed_text = qed_explanation or "No explanation provided."
        else:  # "full"
            similarity_text = similarity_explanation or "No explanation provided."
            qed_text = qed_explanation or "No explanation provided."

        return f"Similarity:\n{similarity_text}\n\nQED:\n{qed_text}"

    def build_feedback(self, state: WorkflowState, result: OracleResult) -> str:
        combined_score = result["Combined"]["value"]

        # Use generic feedback for no_description mode
        if self._xai == "no_description":
            return get_generic_feedback(
                smiles=state["current_smiles"],
                score=combined_score,
                iteration=state["iteration_count"],
                max_iterations=self._max_iterations,
                higher_is_better=True,
            )

        similarity_score = result.get(self.SIMILARITY_KEY, {}).get("value", 0.0)
        qed_score = result.get(self.QED_KEY, {}).get("value", 0.0)

        sim_ok = similarity_score >= self._min_similarity
        qed_ok = qed_score >= self._min_qed
        combined_ok = combined_score >= self._target_score

        # Check if molecule is identical to target
        sim_explanation = result.get(self.SIMILARITY_KEY, {}).get("explanation", "")
        is_identical = "IDENTICAL MOLECULE" in sim_explanation

        explanation = self._get_explanation(result)

        feedback = f"""SMILES: {state["current_smiles"]}
Iteration: {state["iteration_count"]} / {self._max_iterations}

Combined score: {combined_score:.4f} (target: ≥ {self._target_score:.2f}) {"✓" if combined_ok else "✗"}
Similarity: {similarity_score:.4f} (min: {self._min_similarity:.2f}) {"✓" if sim_ok else "✗"}
QED: {qed_score:.4f} (min: {self._min_qed:.2f}) {"✓" if qed_ok else "✗"}

Explanation:

{explanation}"""

        if is_identical:
            feedback += """

WARNING: You proposed the exact target molecule. This is NOT allowed.
You must propose a molecule that is structurally SIMILAR but NOT IDENTICAL to the target."""

        feedback += """

Respond with JSON only:
{
  "reason": "<what you changed and why>",
  "smiles": "<new SMILES>"
}"""
        return feedback.strip()

    def is_done(self, state: WorkflowState, result: OracleResult) -> bool:
        if state["iteration_count"] >= self._max_iterations:
            return True

        sim_explanation = result.get(self.SIMILARITY_KEY, {}).get("explanation", "")
        if "IDENTICAL MOLECULE" in sim_explanation:
            return False

        combined_score = result["Combined"]["value"]
        similarity_score = result.get(self.SIMILARITY_KEY, {}).get("value", 0.0)
        qed_score = result.get(self.QED_KEY, {}).get("value", 0.0)

        # Success: combined score meets target AND both minimums satisfied
        if (
            combined_score >= self._target_score
            and similarity_score >= self._min_similarity
            and qed_score >= self._min_qed
        ):
            return True

        return False

    def max_iterations(self) -> int:
        return self._max_iterations
