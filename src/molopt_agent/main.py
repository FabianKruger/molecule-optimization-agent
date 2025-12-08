from dotenv import load_dotenv

from .cli import parse_args
from .config import load_experiment_config

from langchain_core.runnables import RunnableConfig

from .config import ExperimentConfig
from .state import make_initial_state
from .oracles import build_oracle_from_config
from .objectives import build_objective_from_config
from .graph.builder import build_workflow
from .save_experiment import save_conversation_log


def run_experiment(cfg: ExperimentConfig):
    oracle = build_oracle_from_config(cfg.oracle)
    objective = build_objective_from_config(cfg.objective, oracle=oracle)

    app = build_workflow(objective, cfg)
    initial_state = make_initial_state()

    final_state = app.invoke(
        initial_state,
        config=RunnableConfig(recursion_limit=cfg.recursion_limit),
    )

    log_file = save_conversation_log(
        final_state,
        config=cfg,
        output_dir=cfg.log_dir,
        experiment_name=cfg.experiment_name,
    )

    return final_state, log_file



def main():
    load_dotenv()

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
