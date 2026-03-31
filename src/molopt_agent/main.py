from dotenv import load_dotenv
import logging

from .cli import parse_args
import tempfile

from .config import load_experiment_config

from langchain_core.runnables import RunnableConfig

from .config import ExperimentConfig
from .state import make_initial_state
from .oracles import build_oracle_from_config
from .objectives import build_objective_from_config
from .graph.builder import build_workflow
from .save_experiment import save_conversation_log, plot_trajectory


def setup_logging():
    """Configure logging for the molecule optimization agent."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Reduce noise from external libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('langchain').setLevel(logging.WARNING)


def run_experiment(cfg: ExperimentConfig):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment: {cfg.experiment_name}")
    logger.info(f"Oracle: {cfg.oracle.name}")
    logger.info(f"Objective: {cfg.objective.name}")
    logger.info(f"Max iterations: {cfg.objective.params.get('max_iterations', 'N/A')}")

    oracle = build_oracle_from_config(cfg.oracle)
    objective = build_objective_from_config(cfg.objective, oracle=oracle)

    app = build_workflow(objective, cfg)
    initial_state = make_initial_state()
    final_state = initial_state
    workflow_error = None

    logger.info("Starting optimization workflow...")
    try:
        for event in app.stream(
            initial_state,
            config=RunnableConfig(recursion_limit=cfg.recursion_limit),
            stream_mode="values",
        ):
            final_state = event
    except Exception as error:
        workflow_error = error
        logger.exception("Optimization workflow failed; saving partial state")
        final_state["termination_reason"] = final_state.get("termination_reason") or "workflow_error"
        final_state["generation_error"] = final_state.get("generation_error") or str(error)
        if not final_state.get("final_response"):
            final_state["final_response"] = (
                "Optimization terminated before the final summary completed. "
                f"Error: {type(error).__name__}: {error}"
            )

    if workflow_error is None:
        logger.info(f"Optimization completed after {final_state['iteration_count']} iterations")
    else:
        logger.warning(
            "Optimization stopped after %s iterations due to an error",
            final_state["iteration_count"],
        )

    try:
        log_file, timestamp = save_conversation_log(
            final_state,
            config=cfg,
            output_dir=cfg.log_dir,
            experiment_name=cfg.experiment_name,
        )
    except Exception as save_error:
        logger.error(
            "Primary save to '%s' failed (%s); attempting fallback save",
            cfg.log_dir,
            save_error,
        )
        _fallback_dir = tempfile.gettempdir()
        try:
            log_file, timestamp = save_conversation_log(
                final_state,
                config=cfg,
                output_dir=_fallback_dir,
                experiment_name=cfg.experiment_name,
            )
            logger.warning("Conversation saved to fallback location: %s", log_file)
        except Exception as fallback_error:
            logger.critical(
                "Fallback save also failed (%s). Conversation data could not be persisted.",
                fallback_error,
            )
            log_file = None
            timestamp = "unknown"

    try:
        plot_file = plot_trajectory(
            final_state,
            objective_name=cfg.objective.name,
            output_dir=cfg.log_dir,
            experiment_name=cfg.experiment_name,
            timestamp=timestamp,
            model_name=cfg.llm.model,
        )
        if plot_file:
            logger.info(f"Trajectory plot saved to: {plot_file}")
    except Exception as plot_error:
        logger.warning("Trajectory plot failed (non-fatal): %s", plot_error)

    if workflow_error is not None:
        raise workflow_error

    return final_state, log_file



def main():
    load_dotenv()
    setup_logging()

    args = parse_args()

    try:
        cfg = load_experiment_config(args.config)
    except ValueError as e:
        print(f"Configuration error: {e}")
        raise SystemExit(1)

    final_state, log_file = run_experiment(cfg)

    print("\n=== FINAL RESULTS ===")
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Best SMILES: {final_state['current_smiles']}")
    print(f"Iterations: {final_state['iteration_count']}")
    print(f"Conversation log saved to: {log_file}")
    print(f"\nSummary:\n{final_state['final_response']}")


if __name__ == "__main__":
    main()
