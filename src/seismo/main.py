from dotenv import load_dotenv
import logging

from .cli import parse_args
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

    logger.info("Starting optimization workflow...")
    final_state = app.invoke(
        initial_state,
        config=RunnableConfig(recursion_limit=cfg.recursion_limit),
    )

    logger.info(f"Optimization completed after {final_state['iteration_count']} iterations")
    log_file, timestamp = save_conversation_log(
        final_state,
        config=cfg,
        output_dir=cfg.log_dir,
        experiment_name=cfg.experiment_name,
    )

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
