from typing import Optional

from ..state import WorkflowState
from ..oracles.base import Oracle, OracleResult


FIRST_MESSAGE_TEMPLATE = """
We are optimizing ligands for protein-ligand binding.

Target Protein Sequence:
{protein_sequence}

Objective:
- Maximize the predicted binding probability.
- Target: Binding probability >= {target_binding_probability:.3f}
{additional_info_section}
Step 1:
Propose a single initial molecule as a SMILES string that you expect to have strong binding affinity to the target protein.

Respond with a single JSON object:
{{
  "reason": "<why this is a reasonable starting point for this objective>",
  "smiles": "<SMILES string>"
}}
""".strip()


class Boltz2Objective:
    name = "boltz2"

    def __init__(
        self,
        oracle: Oracle,
        target_binding_probability: float,
        max_iterations: int,
        additional_information: Optional[str] = None,
    ):
        self.oracle = oracle
        self._target_binding_probability = float(target_binding_probability)
        self._max_iterations = int(max_iterations)
        self._additional_information = additional_information

    # --- prompts ---

    def first_message(self) -> str:
        additional_info_section = ""
        if self._additional_information:
            additional_info_section = f"\nAdditional Information:\n{self._additional_information}\n"

        return FIRST_MESSAGE_TEMPLATE.format(
            protein_sequence=self.oracle.protein_sequence,
            target_binding_probability=self._target_binding_probability,
            additional_info_section=additional_info_section,
        )

    # --- core logic ---

    def evaluate(self, state: WorkflowState) -> OracleResult:
        smiles = state["current_smiles"]
        return self.oracle(smiles)

    def build_feedback(self, state: WorkflowState, result: OracleResult) -> str:
        binding_probability = result["score"]
        explanation = result.get("explanation", "")

        explanation_section = ""
        if explanation:
            explanation_section = f"""
Oracle explanation:
{explanation}
"""

        return f"""
Evaluation of your last proposal:

SMILES: {state['current_smiles']}
Predicted Binding Probability: {binding_probability:.3f} (higher is better, range 0-1)
Goal: Binding probability >= {self._target_binding_probability:.3f}
Iteration: {state['iteration_count']} / {self._max_iterations}
{explanation_section}
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

        binding_probability = result["score"]
        if binding_probability is not None and binding_probability >= self._target_binding_probability:
            return True

        return False

    def max_iterations(self) -> int:
        return self._max_iterations
