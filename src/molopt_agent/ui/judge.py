"""LLM-as-Judge for evaluating user constraints on molecules."""

import os
import json
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


JUDGE_SYSTEM_PROMPT = """You are a helpful chemistry assistant acting as a judge. Your job is to evaluate whether a newly proposed molecule satisfies the user's feedback/constraints.

You will be given:
1. The PREVIOUS molecule (before user feedback) with its scores
2. The NEW molecule (after optimization based on feedback) with its scores  
3. A list of user constraints/feedback that the new molecule should satisfy

Your task is to determine if the NEW molecule reasonably satisfies ALL user constraints when compared to the previous molecule.

IMPORTANT GUIDELINES:
- Be LENIENT and give the benefit of the doubt. If it's unclear whether a constraint is satisfied, assume it is.
- For vague feedback (e.g., "make it better", "I don't like it"), be very generous in accepting improvements.
- Focus on whether a good-faith effort was made to address the feedback.
- Chemical modifications often have trade-offs; don't be overly strict.
- If the feedback is ambiguous, interpret it charitably.

Respond with a JSON object:
{
    "satisfied": true/false,
    "reason": "Brief explanation of your judgment"
}

Only output the JSON, no other text."""


@dataclass
class JudgeResult:
    """Result from the LLM judge evaluation."""
    satisfied: bool
    reason: str


@dataclass 
class MoleculeInfo:
    """Information about a molecule for the judge."""
    smiles: str
    scores: dict[str, float]  # e.g., {"Similarity": 0.72, "QED": 0.85, "Combined": 0.78}


class MoleculeJudge:
    """
    LLM-based judge that evaluates if a molecule satisfies user constraints.
    
    Compares the previous molecule (before feedback) with the new molecule
    to determine if user constraints have been addressed.
    """
    
    def __init__(
        self,
        model: str = "gpt-5.1",
        temperature: float = 0.0,
    ):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable")
        
        llm_kwargs = {
            "model": model,
            "api_key": api_key,
            "temperature": temperature,
        }
        
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            llm_kwargs["base_url"] = base_url
        
        self.llm = ChatOpenAI(**llm_kwargs)
    
    def evaluate(
        self,
        previous_molecule: MoleculeInfo,
        new_molecule: MoleculeInfo,
        user_constraints: list[str],
    ) -> JudgeResult:
        """
        Evaluate if the new molecule satisfies user constraints.
        
        Args:
            previous_molecule: The molecule before user feedback
            new_molecule: The new molecule after optimization
            user_constraints: List of accumulated user feedback/constraints
        
        Returns:
            JudgeResult with satisfied (bool) and reason (str)
        """
        if not user_constraints:
            # No constraints to check
            return JudgeResult(satisfied=True, reason="No user constraints to evaluate.")
        
        # Format scores for display
        prev_scores_str = ", ".join(
            f"{k}: {v:.3f}" for k, v in previous_molecule.scores.items()
        )
        new_scores_str = ", ".join(
            f"{k}: {v:.3f}" for k, v in new_molecule.scores.items()
        )
        
        # Format constraints
        constraints_str = "\n".join(
            f"{i+1}. \"{c}\"" for i, c in enumerate(user_constraints)
        )
        
        prompt = f"""## PREVIOUS Molecule (before user feedback)
SMILES: {previous_molecule.smiles}
Scores: {prev_scores_str}

## NEW Molecule (after optimization)
SMILES: {new_molecule.smiles}
Scores: {new_scores_str}

## User Constraints (ALL should be satisfied)
{constraints_str}

Does the NEW molecule satisfy all user constraints compared to the PREVIOUS molecule?
Remember: Be lenient and give the benefit of the doubt."""

        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            # Parse JSON response
            result = json.loads(response.content.strip())
            return JudgeResult(
                satisfied=bool(result.get("satisfied", True)),
                reason=str(result.get("reason", "No reason provided."))
            )
        except (json.JSONDecodeError, KeyError):
            # If parsing fails, be lenient and accept
            return JudgeResult(
                satisfied=True,
                reason="Could not parse judge response; accepting molecule by default."
            )
