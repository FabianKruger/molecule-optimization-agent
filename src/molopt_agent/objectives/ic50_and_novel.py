from typing import Literal

from ..oracles.base import Oracle, OracleResult
from ..state import WorkflowState


FIRST_MESSAGE_TEMPLATE = """
We are optimizing ligands for the SARS-COV-2 Main Protease (MPro).

Objectives:
1. Minimize the predicted half maximal inhibitory concentration IC50 (in nM).
   - Target: IC50 < {target_ic50:.2f} nM.

2. The final molecule must be NOVEL (not present in PubChem).
   - You may propose non-novel molecules during optimization to learn from them.
   - However, the final accepted molecule MUST be novel.

Step 1:
Propose a single initial molecule as a SMILES string that you expect to have strong inhibitory effect on the enzyme.

Respond with a single JSON object:
{{
  "reason": "<why this is a reasonable starting point for this objective>",
  "smiles": "<SMILES string>"
}}
""".strip()


class IC50AndNovelObjective:
    """
    Multi-objective optimization for IC50 AND novelty.
    
    - Score is based on IC50 only (lower is better).
    - Novelty is a hard constraint: the final molecule must be novel (not in PubChem).
    - Non-novel molecules can be proposed during optimization but won't be accepted as final.
    """

    name = "ic50_and_novel"

    IC50_KEY = "IC50"
    NOVELTY_KEY = "Novelty"

    def __init__(
        self,
        oracle: Oracle,
        target_ic50_nM: float = 10.0,
        max_iterations: int = 20,
        xai: Literal["full", "none"] = "full",
    ):
        self.oracle = oracle
        self._target_ic50_nM = float(target_ic50_nM)
        self._max_iterations = int(max_iterations)
        self._xai = xai

    def first_message(self) -> str:
        return FIRST_MESSAGE_TEMPLATE.format(target_ic50=self._target_ic50_nM)

    def evaluate(self, state: WorkflowState) -> OracleResult:
        smiles = state["current_smiles"]
        return self.oracle(smiles)

    def _get_explanation(self, result: OracleResult) -> str:
        """Get explanation based on xai setting."""
        explanations = result["explanations"]

        if self._xai == "none":
            ic50_text = "No explanation provided."
        else:
            ic50_text = explanations[self.IC50_KEY]

        novelty_text = explanations[self.NOVELTY_KEY]

        return f"IC50 prediction:\n{ic50_text}\n\nNovelty:\n{novelty_text}"

    def build_feedback(self, state: WorkflowState, result: OracleResult) -> str:
        scores = result["scores"]

        ic50_score = scores[self.IC50_KEY]
        novelty_score = scores[self.NOVELTY_KEY]

        ic50_ok = ic50_score < self._target_ic50_nM
        is_novel = novelty_score == 1.0

        if is_novel:
            novelty_status = "✓ NOVEL - Not in PubChem"
        else:
            novelty_status = "✗ EXISTS - Found in PubChem"

        explanation = self._get_explanation(result)

        return f"""Evaluation of your last proposal:

SMILES: {state['current_smiles']}
Predicted IC50: {ic50_score:.2f} nM (target: < {self._target_ic50_nM:.2f} nM) {"✓" if ic50_ok else "✗"}
Novelty: {novelty_status}
Iteration: {state['iteration_count']} / {self._max_iterations}

{explanation}

Respond with a single JSON object only:
{{
  "reason": "<short explanation of what you changed and why>",
  "smiles": "<one NEW SMILES string, not identical to any previous one>"
}}

Do not include any additional text, comments, Markdown, or code fences.""".strip()

    def is_done(self, state: WorkflowState, result: OracleResult) -> bool:
        if state["iteration_count"] >= self._max_iterations:
            return True

        scores = result["scores"]
        ic50_score = scores[self.IC50_KEY]
        novelty_score = scores[self.NOVELTY_KEY]

        # Success: IC50 below target AND molecule is novel
        return ic50_score < self._target_ic50_nM and novelty_score == 1.0

    def max_iterations(self) -> int:
        return self._max_iterations
