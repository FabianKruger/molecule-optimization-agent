import json
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for CLI
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .config import ExperimentConfig
from .state import WorkflowState


def save_conversation_log(
    state: WorkflowState,
    config: ExperimentConfig,
    output_dir: str,
    experiment_name: str,
) -> tuple[str, str]:
    """
    Save conversation log to JSON file.

    Returns:
        Tuple of (log_file_path, timestamp)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(output_dir) / f"{experiment_name}_conversation_{ts}.json"

    conversation = [
        {"role": msg.__class__.__name__, "content": msg.content}
        for msg in state["messages"]
    ]

    data = {
        "timestamp": ts,
        "experiment": experiment_name,
        "config": asdict(config),
        "conversation": conversation,
        "trace": state["trace"],
        "iterations": state["iteration_count"],
        "summary": state["final_response"],
    }

    with open(log_file, "w") as f:
        json.dump(data, f, indent=2)

    return str(log_file), ts


def plot_trajectory(
    state: WorkflowState,
    objective_name: str,
    output_dir: str,
    experiment_name: str,
    timestamp: str,
    model_name: str,
) -> str:
    """
    Plot the optimization trajectory showing score vs oracle calls.

    Args:
        state: The final workflow state containing the trace
        objective_name: Name of the objective for the y-axis label
        output_dir: Directory to save the plot
        experiment_name: Name of the experiment for the filename
        timestamp: Timestamp string to use in the filename (format: YYYYMMDD_HHMMSS)
        model_name: Name of the LLM model used

    Returns:
        Path to the saved plot file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_file = Path(output_dir) / f"{experiment_name}_trajectory_{timestamp}.png"

    trace = state["trace"]

    if not trace:
        # No data to plot
        return None

    # Extract iterations and scores from trace
    iterations = [entry["iteration"] for entry in trace]
    scores = [entry["score"] for entry in trace]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, scores, marker='o', linewidth=2, markersize=6)
    ax.set_xlabel("Oracle calls", fontsize=12)
    ax.set_ylabel(f"{objective_name} score", fontsize=12)
    ax.set_title(f"MolOpt Agent Trajectory (Model: {model_name})", fontsize=14, fontweight='bold')

    # Force x-axis to show only integer values
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    return str(plot_file)
