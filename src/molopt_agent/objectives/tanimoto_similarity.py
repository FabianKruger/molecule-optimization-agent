from ..oracles.base import Oracle, OracleResult
from ..state import WorkflowState


FIRST_MESSAGE_TEMPLATE = """
We are optimizing molecules for structural similarity to a target molecule using MACCS fingerprints and Tanimoto similarity.

Target molecule SMILES: {target_smiles}

Objective:
- Maximize Tanimoto similarity to the target (range 0-1, higher is better).
- Target: similarity ≥ {target_similarity:.2f}
- IMPORTANT: You must propose molecules that are STRUCTURALLY DIFFERENT from the target. Proposing the exact same molecule as the target is not allowed.

The oracle evaluates similarity using 166 MACCS keys, which encode presence/absence of specific structural features (functional groups, ring systems, atom types, etc.). 
You will receive feedback on which structural features are missing or extra compared to the target.

Step 1:
Propose a single initial molecule as a SMILES string that you expect to be structurally similar to the target, but NOT identical.

Respond with a single JSON object:
{{
  "reason": "<why this molecule should be similar to the target>",
  "smiles": "<SMILES string - must be different from target>"
}}
""".strip()


class TanimotoSimilarityObjective:
    """Objective for maximizing Tanimoto similarity to a target molecule using MACCS keys."""

    name = "tanimoto_similarity"

    def __init__(
        self, 
        oracle: Oracle, 
        target_similarity: float = 0.9, 
        max_iterations: int = 20,
    ):
        self.oracle = oracle
        self._target_similarity = float(target_similarity)
        self._max_iterations = int(max_iterations)
        
        # Extract target_smiles from oracle
        self._target_smiles = self._extract_target_smiles(oracle)

    def _extract_target_smiles(self, oracle: Oracle) -> str:
        """Extract target_smiles from oracle via get_params()."""
        if not hasattr(oracle, "get_params"):
            raise ValueError("Oracle must implement get_params() for TanimotoSimilarityObjective")
        params = oracle.get_params()
        if "target_smiles" not in params:
            raise ValueError("Oracle must provide 'target_smiles' in get_params()")
        return params["target_smiles"]

    def first_message(self) -> str:
        return FIRST_MESSAGE_TEMPLATE.format(
            target_smiles=self._target_smiles,
            target_similarity=self._target_similarity,
        )

    def evaluate(self, state: WorkflowState) -> OracleResult:
        smiles = state["current_smiles"]
        return self.oracle(smiles)

    def build_feedback(self, state: WorkflowState, result: OracleResult) -> str:
        similarity = result["score"]
        explanation = result["explanation"]

        # Check if molecule is identical to target
        is_identical = "IDENTICAL MOLECULE" in explanation

        feedback = f"""
Evaluation of your proposal:

SMILES: {state['current_smiles']}
Tanimoto Similarity: {similarity:.4f}
Goal: achieve similarity ≥ {self._target_similarity:.2f}
Iteration: {state['iteration_count']} / {self._max_iterations}

{explanation}
"""

        if is_identical:
            feedback += """
WARNING: You proposed the exact target molecule. This is NOT allowed.
The similarity score was set to 0.0 because identical molecules are not accepted.
You must propose a molecule that is structurally SIMILAR but NOT IDENTICAL to the target.
"""

        feedback += """
Based on the MACCS key differences above, propose a modified molecule that:
1. Adds missing structural features (if any)
2. Removes extra structural features (if any)
3. Is NOT identical to the target molecule

Respond with JSON only:
{
  "reason": "<what structural changes you made and why>",
  "smiles": "<new SMILES - must be different from both target and previous proposals>"
}
"""
        return feedback.strip()

    def is_done(self, state: WorkflowState, result: OracleResult) -> bool:
        # Hard iteration cap
        if state["iteration_count"] >= self._max_iterations:
            return True

        # Check for identical molecule (not allowed as success)
        if "IDENTICAL MOLECULE" in result.get("explanation", ""):
            return False

        # Success: similarity meets target (and not identical)
        similarity = result["score"]
        if similarity is not None and similarity >= self._target_similarity:
            return True

        return False

    def max_iterations(self) -> int:
        return self._max_iterations

