"""
Generic messages for no_description mode.

These messages are used when xai="no_description" to hide task-specific details
from the agent, allowing us to test if the agent can optimize without knowing
what the objective actually measures.
"""

GENERIC_FIRST_MESSAGE_TEMPLATE = """
Your task is to optimize the score. {direction}.

Propose a single molecule as a SMILES string.

Respond with a single JSON object:
{{
  "reason": "<why this is a reasonable starting point>",
  "smiles": "<SMILES string>"
}}
""".strip()


GENERIC_FEEDBACK_TEMPLATE = """
SMILES: {smiles}
Score: {score:.4f} ({direction})
Iteration: {iteration} / {max_iterations}

Respond with JSON only:
{{
  "reason": "<what you changed and why>",
  "smiles": "<new SMILES>"
}}
""".strip()


def get_generic_first_message(higher_is_better: bool = True) -> str:
    """Get the generic first message for no_description mode."""
    direction = "Higher is better" if higher_is_better else "Lower is better"
    return GENERIC_FIRST_MESSAGE_TEMPLATE.format(direction=direction)


def get_generic_feedback(
    smiles: str,
    score: float,
    iteration: int,
    max_iterations: int,
    higher_is_better: bool = True,
) -> str:
    """Get generic feedback message for no_description mode."""
    direction = "higher is better" if higher_is_better else "lower is better"
    return GENERIC_FEEDBACK_TEMPLATE.format(
        smiles=smiles,
        score=score,
        direction=direction,
        iteration=iteration,
        max_iterations=max_iterations,
    )

