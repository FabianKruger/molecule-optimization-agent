import json
from typing import Optional

from ..oracles.base import Oracle, OracleResult
from ..state import WorkflowState

FIRST_MESSAGE_PROBABILITY = """
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


FIRST_MESSAGE_AFFINITY = """
We are optimizing ligands for protein-ligand binding.

Target Protein Sequence:
{protein_sequence}

Objective:
- Maximize the predicted binding affinity (higher values indicate stronger predicted binding)
- Constraint: Binding probability should be > {target_binding_probability}
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

        # Detect which mode we're in based on oracle's binding_score_name
        binding_score_name = getattr(
            oracle, "binding_score_name", "affinity_probability_binary"
        )
        self._is_probability_mode = binding_score_name == "affinity_probability_binary"

    # --- prompts ---

    def first_message(self) -> str:
        additional_info_section = ""
        if self._additional_information:
            additional_info_section = (
                f"\nAdditional Information:\n{self._additional_information}\n"
            )

        template = (
            FIRST_MESSAGE_PROBABILITY
            if self._is_probability_mode
            else FIRST_MESSAGE_AFFINITY
        )
        return template.format(
            protein_sequence=self.oracle.protein_sequence,
            target_binding_probability=self._target_binding_probability,
            additional_info_section=additional_info_section,
        )

    # --- core logic ---

    def evaluate(self, state: WorkflowState) -> OracleResult:
        smiles = state["current_smiles"]
        return self.oracle(smiles)

    def build_feedback(self, state: WorkflowState, result: OracleResult) -> str:
        explanation = result.get("explanation", "")
        extra_scores_json = result.get("extra_scores", "{}")
        extra_scores = json.loads(extra_scores_json)

        explanation_section = ""
        if explanation:
            explanation_section = f"""
Oracle explanation:
{explanation}
"""

        if self._is_probability_mode:
            # Probability mode: optimize binding probability
            binding_probability = result["score"]
            return f"""
Evaluation of your last proposal:

SMILES: {state["current_smiles"]}
Predicted Binding Probability: {binding_probability:.3f} (higher is better, range 0-1)
Goal: Binding probability >= {self._target_binding_probability:.3f}
Iteration: {state["iteration_count"]} / {self._max_iterations}
{explanation_section}
Use this evaluation together with the full history of molecules and evaluations above to decide on the next molecule.

Respond with a single JSON object only, with this exact structure:
{{
  "reason": "<short explanation>",
  "smiles": "<one NEW SMILES string, not identical to any previous one>"
}}

Do not include any additional text, comments, Markdown, or code fences.
""".strip()
        else:
            # Affinity mode: optimize affinity with probability constraint
            affinity = result["score"]
            # parse probability from oracle explanation
            binding_probability = extra_scores.get("affinity_probability_binary", None)

            probability_info = ""
            if binding_probability is not None:
                probability_info = f"\nBinding Probability: {binding_probability:.3f} (constraint: > {self._target_binding_probability})"

            return f"""
Evaluation of your last proposal:

SMILES: {state["current_smiles"]}
Predicted Binding Affinity: {affinity:.3f} (higher values indicate stronger predicted binding) {probability_info}
Iteration: {state["iteration_count"]} / {self._max_iterations}
{explanation_section}
Use this evaluation together with the full history of molecules and evaluations above to decide on the next molecule.
Remember: Optimize for the highest binding affinity while keeping binding probability > {self._target_binding_probability}

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

        if self._is_probability_mode:
            # Probability mode: check if target probability is reached
            binding_probability = result["score"]
            if (
                binding_probability is not None
                and binding_probability >= self._target_binding_probability
            ):
                return True
        # In affinity mode, we only stop at max_iterations since we're continuously for higher values

        return False

    def max_iterations(self) -> int:
        return self._max_iterations
