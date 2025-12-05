from typing import Optional

from ..state import WorkflowState
from ..oracles.base import Oracle, OracleResult
from .base import Objective


FIRST_MESSAGE_TEMPLATE = """
We are optimizing ligands for the SARS-COV-2 Main Protease (MPro).

Objective:
- Minimize the predicted inhibitory constant IC50 (in nM).
- Target: IC50 < {target_ic50:.2f} nM.

Step 1:
Propose a single initial molecule as a SMILES string that you expect to have strong inhibitory effect on the enzyme.

Respond with a single JSON object:
{{
  "reason": "<why this is a reasonable starting point for this objective>",
  "smiles": "<SMILES string>"
}}
""".strip()


class XGBoostSARSCoV2Objective:
    name = "xgboost_sarscov2"

    def __init__(self, oracle: Oracle, target_ic50_nM: float, max_iterations: int):
        self.oracle = oracle
        self._target_ic50_nM = float(target_ic50_nM)
        self._max_iterations = int(max_iterations)

    # --- prompts ---

    def first_message(self) -> str:
        return FIRST_MESSAGE_TEMPLATE.format(target_ic50=self._target_ic50_nM)

    # --- core logic ---

    def evaluate(self, state: WorkflowState) -> OracleResult:
        smiles = state["current_smiles"]
        return self.oracle(smiles)

    def build_feedback(self, state: WorkflowState, result: OracleResult) -> str:
        ic50 = result["score"]
        explanation = result["explanation"]

        return f"""
Evaluation of your last proposal:

SMILES: {state['current_smiles']}
Predicted IC50: {ic50:.2f} nM (lower is better)
Goal: IC50 < {self._target_ic50_nM:.2f} nM.
Iteration: {state['iteration_count']} / {self._max_iterations}

Oracle explanation:
{explanation}

Use this evaluation together with the full history of molecules and evaluations above to decide on the next molecule.

Respond with a single JSON object only, with this exact structure:
{{
  "reason": "<short explanation>",
  "smiles": "<one NEW SMILES string, not identical to any previous one>"
}}

Do not include any additional text, comments, Markdown, or code fences.
""".strip()

    def is_done(self, state: WorkflowState, result: OracleResult) -> bool:
        # Hard iteration cap
        if state["iteration_count"] >= self._max_iterations:
            return True

        ic50 = result["score"]
        if ic50 is not None and ic50 < self._target_ic50_nM:
            return True

        return False

    def max_iterations(self) -> int:
        return self._max_iterations
