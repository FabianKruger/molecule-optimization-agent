import json
from dataclasses import asdict
from pathlib import Path
from datetime import datetime

from .config import ExperimentConfig
from .state import WorkflowState


def save_conversation_log(
    state: WorkflowState,
    config: ExperimentConfig,
    output_dir: str,
    experiment_name: str,
) -> str:
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

    return str(log_file)
