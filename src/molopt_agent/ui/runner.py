"""Interactive session manager for human-in-the-loop molecule optimization."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from PIL import Image

from ..config import ExperimentConfig, LLMConfig, OracleConfig, ObjectiveConfig
from ..state import WorkflowState, make_initial_state
from ..oracles import build_oracle_from_config
from ..objectives import build_objective_from_config
from ..objectives.base import Objective
from ..oracles.base import OracleResult
from ..graph.builder import build_workflow

from .judge import MoleculeJudge, MoleculeInfo, JudgeResult
from .mol_utils import smiles_to_image


# Default Quercetin SMILES
DEFAULT_TARGET_SMILES = "O=C1c3c(O/C(=C1/O)c2ccc(O)c(O)c2)cc(O)cc3O"

# Max iterations for post-feedback optimization rounds
POST_FEEDBACK_MAX_ITERATIONS = 5


@dataclass
class TraceEntry:
    """A single entry in the optimization trace."""
    iteration: int
    smiles: str
    reason: str
    scores: dict[str, float] | None  # None if invalid SMILES
    combined_score: float | None
    is_valid: bool
    image: Image.Image | None = None


@dataclass
class SessionResult:
    """Result from an optimization session."""
    smiles: str
    scores: dict[str, float]
    combined_score: float
    summary: str
    image: Image.Image | None
    iteration_count: int
    trace: list[TraceEntry] = field(default_factory=list)
    judge_result: JudgeResult | None = None


class ObjectiveWrapper:
    """
    Wrapper around an objective that allows controlling termination.
    
    Used to:
    1. Force continuation even when original objectives are met (for user feedback rounds)
    2. Limit max iterations for feedback rounds
    """
    
    def __init__(self, wrapped: Objective, override_max_iterations: int | None = None):
        self._wrapped = wrapped
        self._override_max_iterations = override_max_iterations
        self._force_continue = False
        self._continuation_iteration_start = 0
        self.name = wrapped.name
    
    def first_message(self) -> str:
        return self._wrapped.first_message()
    
    def evaluate(self, state: WorkflowState) -> OracleResult:
        return self._wrapped.evaluate(state)
    
    def build_feedback(self, state: WorkflowState, result: OracleResult) -> str:
        return self._wrapped.build_feedback(state, result)
    
    def is_done(self, state: WorkflowState, result: OracleResult) -> bool:
        # Check continuation iteration limit if in continuation mode
        if self._force_continue:
            iterations_since_continue = state["iteration_count"] - self._continuation_iteration_start
            if iterations_since_continue >= POST_FEEDBACK_MAX_ITERATIONS:
                return True
        
        # Delegate to wrapped objective
        return self._wrapped.is_done(state, result)
    
    def max_iterations(self) -> int:
        if self._override_max_iterations is not None:
            return self._override_max_iterations
        return self._wrapped.max_iterations()
    
    def start_continuation(self, current_iteration: int):
        """Mark that we're continuing from user feedback."""
        self._force_continue = True
        self._continuation_iteration_start = current_iteration
    
    def reset_continuation(self):
        """Reset continuation mode."""
        self._force_continue = False
        self._continuation_iteration_start = 0


class InteractiveSession:
    """
    Manages a human-in-the-loop molecule optimization session.
    
    Handles:
    - Initial optimization run with streaming
    - User feedback injection
    - Continuation of optimization with accumulated constraints
    - LLM judge evaluation
    - History navigation
    - Conversation saving
    """
    
    def __init__(
        self,
        target_smiles: str = DEFAULT_TARGET_SMILES,
        target_score: float = 0.75,
        min_similarity: float = 0.7,
        min_qed: float = 0.7,
        max_iterations: int = 10,
    ):
        self.target_smiles = target_smiles
        self.target_score = target_score
        self.min_similarity = min_similarity
        self.min_qed = min_qed
        self.max_iterations = max_iterations
        
        # Session state
        self.state: WorkflowState | None = None
        self.accumulated_constraints: list[str] = []
        self.previous_molecule: MoleculeInfo | None = None  # Molecule before user feedback
        self.trace: list[TraceEntry] = []  # Full trace for navigation
        self.current_trace_index: int = -1  # Current position in trace
        
        # Components (initialized on first run)
        self._config: ExperimentConfig | None = None
        self._objective: ObjectiveWrapper | None = None
        self._app = None
        self._judge: MoleculeJudge | None = None
    
    def _build_config(self) -> ExperimentConfig:
        """Build experiment config from session parameters."""
        return ExperimentConfig(
            experiment_name="interactive_similarity_qed",
            llm=LLMConfig(
                model="gpt-4o",  # Good balance of quality/speed for interactive use
                temperature=0.3,
            ),
            recursion_limit=150,
            log_dir="data/runs",
            oracle=OracleConfig(
                name="composite",
                params={
                    "oracles": [
                        {
                            "name": "similarity",
                            "params": {"target_smiles": self.target_smiles}
                        },
                        {
                            "name": "qed",
                            "params": {}
                        }
                    ],
                    "weights": [0.5, 0.5],
                    "names": ["Similarity", "QED"]
                }
            ),
            objective=ObjectiveConfig(
                name="similarity_qed",
                params={
                    "target_score": self.target_score,
                    "min_similarity": self.min_similarity,
                    "min_qed": self.min_qed,
                    "max_iterations": self.max_iterations,
                    "xai": "full",
                }
            )
        )
    
    def _initialize(self):
        """Initialize workflow components."""
        self._config = self._build_config()
        
        oracle = build_oracle_from_config(self._config.oracle)
        base_objective = build_objective_from_config(self._config.objective, oracle=oracle)
        
        # Wrap the objective for controllable termination
        self._objective = ObjectiveWrapper(base_objective)
        
        self._app = build_workflow(self._objective, self._config)
        self._judge = MoleculeJudge()
    
    def _extract_scores(self, state: WorkflowState) -> dict[str, float]:
        """Extract individual scores from oracle result."""
        result = state.get("oracle_result", {})
        scores = result.get("scores", {})
        return {
            "Similarity": scores.get("Similarity", 0.0),
            "QED": scores.get("QED", 0.0),
            "Combined": result.get("score", 0.0),
        }
    
    def _create_trace_entry(self, state: WorkflowState) -> TraceEntry:
        """Create a trace entry from the current state."""
        smiles = state.get("current_smiles", "")
        is_valid = state.get("is_valid", False) and smiles != ""
        
        if is_valid:
            scores = self._extract_scores(state)
            return TraceEntry(
                iteration=state["iteration_count"],
                smiles=smiles,
                reason=state.get("current_reason", ""),
                scores={"Similarity": scores["Similarity"], "QED": scores["QED"]},
                combined_score=scores["Combined"],
                is_valid=True,
                image=smiles_to_image(smiles),
            )
        else:
            return TraceEntry(
                iteration=state["iteration_count"],
                smiles=smiles,
                reason=state.get("current_reason", ""),
                scores=None,
                combined_score=None,
                is_valid=False,
                image=None,
            )
    
    def _create_result(self, state: WorkflowState, judge_result: JudgeResult | None = None) -> SessionResult:
        """Create a SessionResult from the current state."""
        smiles = state["current_smiles"]
        scores = self._extract_scores(state)
        
        return SessionResult(
            smiles=smiles,
            scores={"Similarity": scores["Similarity"], "QED": scores["QED"]},
            combined_score=scores["Combined"],
            summary=state.get("final_response", ""),
            image=smiles_to_image(smiles),
            iteration_count=state["iteration_count"],
            trace=self.trace.copy(),
            judge_result=judge_result,
        )
    
    def start_streaming(self) -> Generator[TraceEntry | SessionResult, None, None]:
        """
        Run initial optimization with streaming updates.
        
        Yields:
            TraceEntry for each iteration, then final SessionResult
        """
        # Initialize components
        self._initialize()
        
        # Create initial state
        self.state = make_initial_state()
        self.trace = []
        
        # Stream through the workflow
        for event in self._app.stream(
            self.state,
            config=RunnableConfig(recursion_limit=self._config.recursion_limit),
            stream_mode="values",
        ):
            self.state = event
            
            # Check if we have a new molecule proposal (after prediction node)
            if event.get("current_smiles") and event.get("iteration_count", 0) > len(self.trace):
                entry = self._create_trace_entry(event)
                self.trace.append(entry)
                self.current_trace_index = len(self.trace) - 1
                yield entry
        
        # Store this molecule as the "previous" for potential user feedback
        self.previous_molecule = MoleculeInfo(
            smiles=self.state["current_smiles"],
            scores=self._extract_scores(self.state),
        )
        
        # Yield final result
        yield self._create_result(self.state)
    
    def start(self) -> SessionResult:
        """
        Run initial optimization (non-streaming version).
        
        Returns:
            SessionResult with the optimized molecule
        """
        result = None
        for item in self.start_streaming():
            if isinstance(item, SessionResult):
                result = item
        return result
    
    def continue_streaming(self, user_feedback: str) -> Generator[TraceEntry | SessionResult, None, None]:
        """
        Continue optimization with user feedback, streaming updates.
        
        Args:
            user_feedback: User's feedback/constraint
        
        Yields:
            TraceEntry for each iteration, then final SessionResult
        """
        if self.state is None or self._app is None:
            raise RuntimeError("Must call start() before continue_with_feedback()")
        
        # Add to accumulated constraints
        self.accumulated_constraints.append(user_feedback)
        
        # Build the feedback message to inject
        constraints_str = "\n".join(
            f"- {c}" for c in self.accumulated_constraints
        )
        
        feedback_message = f"""The user has provided additional feedback on your proposed molecule.

USER FEEDBACK: "{user_feedback}"

You must now propose a NEW molecule that:
1. Still meets all original objectives (Combined score ≥ {self.target_score}, Similarity ≥ {self.min_similarity}, QED ≥ {self.min_qed})
2. Addresses ALL accumulated user requirements:
{constraints_str}

Continue optimizing. Respond with JSON:
{{
  "reason": "<explain how this addresses the user's feedback while meeting objectives>",
  "smiles": "<new SMILES>"
}}"""

        # Inject feedback into conversation
        self.state["messages"].append(HumanMessage(content=feedback_message))
        
        # Reset final_response to allow continuation
        self.state["final_response"] = ""
        
        # Mark objective wrapper for continuation mode
        self._objective.start_continuation(self.state["iteration_count"])
        
        # Re-run the workflow with streaming
        self._app = build_workflow(self._objective, self._config)
        
        iteration_before = len(self.trace)
        
        for event in self._app.stream(
            self.state,
            config=RunnableConfig(recursion_limit=self._config.recursion_limit),
            stream_mode="values",
        ):
            self.state = event
            
            # Check if we have a new molecule proposal
            if event.get("current_smiles") and event.get("iteration_count", 0) > len(self.trace):
                entry = self._create_trace_entry(event)
                self.trace.append(entry)
                self.current_trace_index = len(self.trace) - 1
                yield entry
        
        # Evaluate with judge
        new_molecule = MoleculeInfo(
            smiles=self.state["current_smiles"],
            scores=self._extract_scores(self.state),
        )
        
        judge_result = self._judge.evaluate(
            previous_molecule=self.previous_molecule,
            new_molecule=new_molecule,
            user_constraints=self.accumulated_constraints,
        )
        
        # Update previous molecule for next round
        self.previous_molecule = new_molecule
        
        # Yield final result
        yield self._create_result(self.state, judge_result)
    
    def continue_with_feedback(self, user_feedback: str) -> SessionResult:
        """
        Continue optimization with user feedback (non-streaming version).
        
        Args:
            user_feedback: User's feedback/constraint
        
        Returns:
            SessionResult with the new optimized molecule
        """
        result = None
        for item in self.continue_streaming(user_feedback):
            if isinstance(item, SessionResult):
                result = item
        return result
    
    def get_trace_entry(self, index: int) -> TraceEntry | None:
        """Get a specific trace entry by index."""
        if 0 <= index < len(self.trace):
            return self.trace[index]
        return None
    
    def navigate_previous(self) -> TraceEntry | None:
        """Navigate to previous molecule in trace."""
        if self.current_trace_index > 0:
            self.current_trace_index -= 1
            return self.trace[self.current_trace_index]
        return None
    
    def navigate_next(self) -> TraceEntry | None:
        """Navigate to next molecule in trace."""
        if self.current_trace_index < len(self.trace) - 1:
            self.current_trace_index += 1
            return self.trace[self.current_trace_index]
        return None
    
    def get_current_entry(self) -> TraceEntry | None:
        """Get current trace entry."""
        if 0 <= self.current_trace_index < len(self.trace):
            return self.trace[self.current_trace_index]
        return None
    
    def get_trace_length(self) -> int:
        """Get total number of entries in trace."""
        return len(self.trace)
    
    def get_accumulated_constraints(self) -> list[str]:
        """Get list of accumulated user constraints."""
        return self.accumulated_constraints.copy()
    
    def save_conversation(self, output_dir: str = "data/results/conversations") -> str:
        """
        Save the conversation to a JSON file.
        
        Args:
            output_dir: Directory to save the conversation
        
        Returns:
            Path to the saved file
        """
        if self.state is None:
            raise RuntimeError("No conversation to save. Run optimization first.")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interactive_session_{ts}.json"
        filepath = Path(output_dir) / filename
        
        # Prepare conversation data
        conversation = [
            {"role": msg.__class__.__name__, "content": msg.content}
            for msg in self.state["messages"]
        ]
        
        # Prepare trace data (without images)
        trace_data = [
            {
                "iteration": entry.iteration,
                "smiles": entry.smiles,
                "reason": entry.reason,
                "scores": entry.scores,
                "combined_score": entry.combined_score,
                "is_valid": entry.is_valid,
            }
            for entry in self.trace
        ]
        
        data = {
            "timestamp": ts,
            "experiment": "interactive_similarity_qed",
            "parameters": {
                "target_smiles": self.target_smiles,
                "target_score": self.target_score,
                "min_similarity": self.min_similarity,
                "min_qed": self.min_qed,
                "max_iterations": self.max_iterations,
            },
            "accumulated_constraints": self.accumulated_constraints,
            "conversation": conversation,
            "trace": trace_data,
            "iterations": self.state["iteration_count"],
            "final_smiles": self.state["current_smiles"],
            "summary": self.state.get("final_response", ""),
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)
    
    def reset(self):
        """Reset the session to start fresh."""
        self.state = None
        self.accumulated_constraints = []
        self.previous_molecule = None
        self.trace = []
        self.current_trace_index = -1
        self._config = None
        self._objective = None
        self._app = None
        # Keep judge instance
