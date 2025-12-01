from ..oracles.base import Oracle, OracleResult
from ..state import WorkflowState


FIRST_MESSAGE_TEMPLATE = """
We are optimizing molecules for drug-likeness using the QED (Quantitative Estimate of Drug-likeness) score.

Objective:
- Maximize the QED score (range 0-1, higher is better).
- Target: QED ≥ {target_qed:.2f}

QED evaluates 8 molecular properties:
1. Molecular Weight (optimal: 281-332 Da)
2. LogP/lipophilicity (optimal: 1.95-3.57)
3. Hydrogen Bond Acceptors (optimal: 2-3)
4. Hydrogen Bond Donors (optimal: 1)
5. Polar Surface Area (optimal: 40-69 Ų)
6. Rotatable Bonds (optimal: 2-4)
7. Aromatic Rings (optimal: 1-2)
8. Structural Alerts (optimal: 0 - no problematic substructures)

Step 1:
Propose a single initial molecule as a SMILES string that you expect to have good drug-like properties.

Respond with a single JSON object:
{{
  "reason": "<why this is a reasonable starting point for maximizing QED>",
  "smiles": "<SMILES string>"
}}
""".strip()


class QedObjective:
    """Objective for maximizing QED (drug-likeness) score."""

    name = "qed"

    def __init__(self, oracle: Oracle, target_qed: float = 0.9, max_iterations: int = 20):
        self.oracle = oracle
        self._target_qed = float(target_qed)
        self._max_iterations = int(max_iterations)

    def first_message(self) -> str:
        return FIRST_MESSAGE_TEMPLATE.format(target_qed=self._target_qed)

    def evaluate(self, state: WorkflowState) -> OracleResult:
        smiles = state["current_smiles"]
        return self.oracle(smiles)

    def build_feedback(self, state: WorkflowState, result: OracleResult) -> str:
        qed_score = result["score"]
        explanation = result["explanation"]

        return f"""
SMILES: {state['current_smiles']}
QED: {qed_score:.3f} / {self._target_qed:.2f} (target)
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

        # Success: QED meets target
        qed_score = result["score"]
        if qed_score is not None and qed_score >= self._target_qed:
            return True

        return False

    def max_iterations(self) -> int:
        return self._max_iterations

