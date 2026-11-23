from typing import Literal

from ..oracles.base import Oracle, OracleResult
from ..state import WorkflowState
from .generic_messages import get_generic_first_message, get_generic_feedback


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


def omitted_explanation_text(policy: str, k: int) -> str:
    """Placeholder shown instead of an explanation outside the shortened history."""
    if policy == "last_k":
        reason = f"this molecule is not among the last {k} valid molecules"
    else:
        reason = (
            f"this molecule is not among the {k} best or "
            f"{k} worst valid molecules by combined score"
        )
    return f"[Explanation omitted: {reason}.]"


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
        xai: Literal[
            "full",
            "partial",
            "none",
            "no_description",
            "last_k",
            "best_worst_k",
            "truncated_last_k",
            "truncated_best_worst_k",
        ] = "full",
        history_k: int = 2,
    ):
        self.oracle = oracle
        self._target_score = float(target_score)
        self._min_similarity = float(min_similarity)
        self._min_qed = float(min_qed)
        self._max_iterations = int(max_iterations)
        self._xai = xai
        self.history_k = int(history_k)
        if self.history_k < 1:
            raise ValueError("history_k must be >= 1")

        # Extract from oracle
        oracle_params = self._extract_from_oracle(oracle)
        self._target_smiles = oracle_params["target_smiles"]
        self._similarity_weight = float(oracle_params["weights"][0])
        self._qed_weight = float(oracle_params["weights"][1])

    def _extract_from_oracle(self, oracle: Oracle) -> dict:
        """Extract shared params from oracle via get_params()."""
        if not hasattr(oracle, "get_params"):
            raise ValueError("Oracle must implement get_params() for SimilarityQedObjective")
        
        params = oracle.get_params()
        extracted = {}
        
        # Get weights from composite oracle
        if "weights" not in params:
            raise ValueError("Oracle must provide 'weights' in get_params()")
        extracted["weights"] = params["weights"]
        
        # Check sub-oracles for target_smiles
        for sub_oracle in params.get("oracles", []):
            if hasattr(sub_oracle, "get_params"):
                sub_params = sub_oracle.get_params()
                if "target_smiles" in sub_params:
                    extracted["target_smiles"] = sub_params["target_smiles"]
                    break
        
        if "target_smiles" not in extracted:
            raise ValueError("Could not extract target_smiles from oracle's sub-oracles")
        
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

    @property
    def history_policy(self) -> str:
        """How much explanation history the LLM sees (see graph/history.py)."""
        if self._xai in ("last_k", "best_worst_k"):
            return self._xai
        if self._xai in ("truncated_last_k", "truncated_best_worst_k"):
            return self._xai.removeprefix("truncated_")
        return "full"

    @property
    def history_form(self) -> str:
        """"compact" keeps all iterations (explanations replaced), "truncated" removes iterations."""
        return "truncated" if self._xai.startswith("truncated_") else "compact"

    def _get_explanation(self, result: OracleResult) -> str:
        """Get explanation based on xai setting."""
        explanations = result.get("explanations", {})
        similarity_explanation = explanations.get(self.SIMILARITY_KEY, "")
        qed_explanation = explanations.get(self.QED_KEY, "")

        if self._xai == "none":
            similarity_text = "No explanation provided."
            qed_text = "No explanation provided."
        elif self._xai == "partial":
            similarity_text = "No explanation provided."
            qed_text = qed_explanation or "No explanation provided."
        else:  # "full", "last_k", "best_worst_k" (history is shortened elsewhere)
            similarity_text = similarity_explanation or "No explanation provided."
            qed_text = qed_explanation or "No explanation provided."

        return f"Similarity:\n{similarity_text}\n\nQED:\n{qed_text}"

    def build_feedback(self, state: WorkflowState, result: OracleResult) -> str:
        return self._render_feedback(state, result, self._get_explanation(result))

    def build_compact_feedback(self, state: WorkflowState, result: OracleResult) -> str:
        """Feedback with the explanations replaced by a placeholder."""
        return self._render_feedback(
            state, result, omitted_explanation_text(self._xai, self.history_k)
        )

    def _render_feedback(
        self, state: WorkflowState, result: OracleResult, explanation: str
    ) -> str:
        combined_score = result["score"]

        # Use generic feedback for no_description mode
        if self._xai == "no_description":
            return get_generic_feedback(
                smiles=state["current_smiles"],
                score=combined_score,
                iteration=state["iteration_count"],
                max_iterations=self._max_iterations,
                higher_is_better=True,
            )

        scores = result.get("scores", {})

        similarity_score = scores.get(self.SIMILARITY_KEY, 0.0)
        qed_score = scores.get(self.QED_KEY, 0.0)

        sim_ok = similarity_score >= self._min_similarity
        qed_ok = qed_score >= self._min_qed
        combined_ok = combined_score >= self._target_score

        # Check if molecule is identical to target
        is_identical = "IDENTICAL MOLECULE" in result["explanation"]

        feedback = f"""SMILES: {state['current_smiles']}
Iteration: {state['iteration_count']} / {self._max_iterations}

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

        if "IDENTICAL MOLECULE" in result.get("explanation", ""):
            return False

        combined_score = result["score"]
        scores = result.get("scores", {})
        similarity_score = scores.get(self.SIMILARITY_KEY, 0.0)
        qed_score = scores.get(self.QED_KEY, 0.0)

        # Success: combined score meets target AND both minimums satisfied
        if (combined_score >= self._target_score 
            and similarity_score >= self._min_similarity 
            and qed_score >= self._min_qed):
            return True

        return False

    def max_iterations(self) -> int:
        return self._max_iterations
