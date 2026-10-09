from ..oracles.base import Oracle, OracleResult
from ..state import WorkflowState


FIRST_MESSAGE_TEMPLATE = """
We are generating novel molecules that do not already exist in PubChem.

Objective:
- Propose a molecule that is NOT present in the PubChem database.
- The molecule must be chemically valid (valid SMILES).

Step 1:
Propose a single molecule as a SMILES string that you believe might be novel (not in PubChem).

Respond with a single JSON object:
{{
  "reason": "<why you think this molecule might be novel>",
  "smiles": "<SMILES string>"
}}
""".strip()


class NovelObjective:
    """Objective for generating novel molecules not present in PubChem."""

    name = "novel"

    def __init__(self, oracle: Oracle, max_iterations: int = 20):
        self.oracle = oracle
        self._max_iterations = int(max_iterations)

    def first_message(self) -> str:
        return FIRST_MESSAGE_TEMPLATE

    def evaluate(self, state: WorkflowState) -> OracleResult:
        smiles = state["current_smiles"]
        return self.oracle(smiles)

    def build_feedback(self, state: WorkflowState, result: OracleResult) -> str:
        score = result["score"]
        explanation = result["explanation"]

        if score == 1.0:
            status = "NOVEL - Not in PubChem"
        else:
            status = "EXISTS - Found in PubChem"

        return f"""
SMILES: {state['current_smiles']}
Status: {status}
Iteration: {state['iteration_count']} / {self._max_iterations}

{explanation}

Respond with JSON only:
{{
  "reason": "<what you changed and why>",
  "smiles": "<new SMILES>"
}}
""".strip()

    def is_done(self, state: WorkflowState, result: OracleResult) -> bool:
        # Hard iteration cap
        if state["iteration_count"] >= self._max_iterations:
            return True

        # Success: molecule is novel (not in PubChem)
        if result["score"] == 1.0:
            return True

        return False

    def max_iterations(self) -> int:
        return self._max_iterations
