"""
TDC Top-50 Molecules Baseline Experiment

This script runs multiple optimization models from the mol-opt package on all TDC oracles
with max_oracle_calls=50 to get top-50 molecules for baseline comparison.

Results are saved to: data/results/tdc_top_50_baselines/results.csv

Note: This script needs to be run with an environment built from https://github.com/wenhao-gao/mol-opt
"""
from __future__ import annotations

import csv
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator

# mol-opt imports
from molopt.reinvent import REINVENT  # type: ignore
from molopt.graph_ga import GraphGA  # type: ignore
from molopt.reinvent_selfies import REINVENT_SELFIES  # type: ignore
from molopt.gpbo import GPBO  # type: ignore


# Configuration
MAX_ORACLE_CALLS = 50
FREQ_LOG = 1
PATIENCE = 100  # High patience to avoid early stopping
NUM_REPETITIONS = 5
DEFAULT_SEED = 0

# TDC Oracle names (from run_tdc_tasks_experiments.sh)
TDC_ORACLES = [
    "qed",
    "drd2",
    "gsk3b",
    "jnk3",
    "celecoxib_rediscovery",
    "troglitazone_rediscovery",
    "thiothixene_rediscovery",
    "albuterol_similarity",
    "mestranol_similarity",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "median1",
    "median2",
    "osimertinib_mpo",
    "fexofenadine_mpo",
    "ranolazine_mpo",
    "perindopril_mpo",
    "amlodipine_mpo",
    "sitagliptin_mpo",
    "zaleplon_mpo",
    "valsartan_smarts",
    "deco_hop",
    "scaffold_hop",
]

# Model configurations: key -> (class, internal_name_used_in_yaml)
# Some models use different names internally when saving YAML files
MODELS = {
    "reinvent": (REINVENT, "reinvent"),
    "graph_ga": (GraphGA, "graph_ga"),
    "reinvent_selfies": (REINVENT_SELFIES, "reinvent_selfies"),
    "gpbo": (GPBO, "gp_bo"),  # GPBO saves as "gp_bo" internally
}


def get_results_dir() -> Path:
    """Get the base results directory."""
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "results" / "tdc_top_50_baselines"


def run_single_experiment(
    model_name: str,
    model_class: type,
    yaml_name: str,
    oracle_name: str,
    output_dir: Path,
    seed: int,
    repeat: int,
) -> list[dict[str, Any]]:
    """
    Run a single optimization experiment.

    Args:
        model_name: Name of the model (for logging and CSV)
        model_class: The optimizer class from mol-opt
        yaml_name: Name used by the optimizer in YAML filenames
        oracle_name: TDC oracle name (string)
        output_dir: Directory to save results
        seed: Random seed for reproducibility
        repeat: Repetition index (0-based)

    Returns:
        List of row dictionaries for CSV
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Running {model_name} on {oracle_name} (seed={seed})...")

    optimizer = model_class(
        smi_file=None,
        n_jobs=-1,
        max_oracle_calls=MAX_ORACLE_CALLS,
        freq_log=FREQ_LOG,
        output_dir=str(output_dir),
        log_results=True,
    )

    # Clear the mol_buffer to avoid cached scores from previous experiments.
    # This is necessary due to a mutable default argument bug in mol-opt's Oracle class
    # where mol_buffer={} is shared across all Oracle instances.
    optimizer.oracle.mol_buffer.clear()

    start_time = datetime.now()
    optimizer.optimize(oracle=oracle_name, patience=PATIENCE, seed=seed)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    rows = []

    # Load results from the saved yaml file (use yaml_name, not model_name)
    result_file = output_dir / f"results_{yaml_name}_{oracle_name}_{seed}.yaml"
    if result_file.exists():
        with open(result_file) as f:
            mol_buffer = yaml.safe_load(f)
            if mol_buffer:
                # Sort by score descending for rank
                sorted_mols = sorted(mol_buffer.items(), key=lambda x: x[1][0], reverse=True)
                for rank, (smiles, (score, oracle_call)) in enumerate(sorted_mols, 1):
                    rows.append({
                        "model": model_name,
                        "task": oracle_name,
                        "repeat": repeat,
                        "iteration": int(oracle_call),
                        "molecule": smiles,
                        "score": float(score),
                        "rank": rank,
                        "duration_seconds": duration,
                        "timestamp": start_time.isoformat(),
                    })

    if rows:
        print(f"    Done! Top score: {rows[0]['score']:.4f}, {len(rows)} molecules")
    else:
        print(f"    No molecules found")

    return rows


def check_experiment_exists(base_dir: Path, model_name: str, yaml_name: str, oracle_name: str, repeat: int) -> bool:
    """Check if an experiment already has results (YAML file exists with data).
    
    Args:
        model_name: Directory name for the model
        yaml_name: Name used in YAML filename (may differ from model_name)
    """
    output_dir = base_dir / model_name / oracle_name / f"repeat_{repeat}"
    yaml_files = list(output_dir.glob(f"results_{yaml_name}_{oracle_name}_*.yaml"))
    if not yaml_files:
        return False
    # Check if the file has actual content
    for yaml_file in yaml_files:
        if yaml_file.stat().st_size > 0:
            return True
    return False


def regenerate_csv_from_yaml(base_dir: Path) -> None:
    """Regenerate the CSV file by scanning all existing YAML result files."""
    csv_file = base_dir / "results.csv"
    fieldnames = [
        "model",
        "task",
        "repeat",
        "iteration",
        "molecule",
        "score",
        "rank",
        "duration_seconds",
        "timestamp",
    ]

    all_rows: list[dict[str, Any]] = []

    print("\nRegenerating CSV from YAML files...")
    
    for model_name, (_, yaml_name) in MODELS.items():
        model_dir = base_dir / model_name
        if not model_dir.exists():
            continue
        
        for oracle_name in TDC_ORACLES:
            oracle_dir = model_dir / oracle_name
            if not oracle_dir.exists():
                continue
            
            for repeat in range(NUM_REPETITIONS):
                repeat_dir = oracle_dir / f"repeat_{repeat}"
                if not repeat_dir.exists():
                    continue
                
                # Use yaml_name for the file pattern (may differ from model_name)
                yaml_files = list(repeat_dir.glob(f"results_{yaml_name}_{oracle_name}_*.yaml"))
                for yaml_file in yaml_files:
                    try:
                        with open(yaml_file) as f:
                            mol_buffer = yaml.safe_load(f)
                            if mol_buffer:
                                # Sort by score descending for rank
                                sorted_mols = sorted(mol_buffer.items(), key=lambda x: x[1][0], reverse=True)
                                for rank, (smiles, (score, oracle_call)) in enumerate(sorted_mols, 1):
                                    all_rows.append({
                                        "model": model_name,
                                        "task": oracle_name,
                                        "repeat": repeat,
                                        "iteration": int(oracle_call),
                                        "molecule": smiles,
                                        "score": float(score),
                                        "rank": rank,
                                        "duration_seconds": 0,  # Not available from YAML
                                        "timestamp": "",  # Not available from YAML
                                    })
                    except Exception as e:
                        print(f"  Warning: Could not read {yaml_file}: {e}")

    # Write all results to CSV
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"CSV regenerated: {csv_file}")
    print(f"Total rows: {len(all_rows)}")
    
    # Print summary of tasks covered
    tasks_covered = set(row["task"] for row in all_rows)
    tasks_missing = set(TDC_ORACLES) - tasks_covered
    print(f"Tasks covered: {len(tasks_covered)}/{len(TDC_ORACLES)}")
    if tasks_missing:
        print(f"Tasks missing: {tasks_missing}")


def run_all_experiments(rng: Generator) -> None:
    """
    Run all model-oracle combinations and save to CSV.

    Each model-oracle combination is run NUM_REPETITIONS times with different seeds
    generated from the provided random generator. Experiments that already have
    results (YAML files) are skipped.

    Args:
        rng: NumPy random generator for generating experiment seeds.
    """
    base_dir = get_results_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    # First pass: count how many experiments need to run
    experiments_to_run = []
    experiments_skipped = 0
    
    for model_name, (model_class, yaml_name) in MODELS.items():
        for oracle_name in TDC_ORACLES:
            for repeat in range(NUM_REPETITIONS):
                # Generate a unique seed for this repetition (must be deterministic)
                seed = int(rng.integers(0, 2**31))
                
                if check_experiment_exists(base_dir, model_name, yaml_name, oracle_name, repeat):
                    experiments_skipped += 1
                else:
                    experiments_to_run.append((model_name, model_class, yaml_name, oracle_name, repeat, seed))
    
    total_experiments = len(experiments_to_run)
    print(f"\nExperiments to run: {total_experiments}")
    print(f"Experiments skipped (already have data): {experiments_skipped}")
    
    if total_experiments == 0:
        print("No new experiments to run.")
    else:
        # Run missing experiments
        for current, (model_name, model_class, yaml_name, oracle_name, repeat, seed) in enumerate(experiments_to_run, 1):
            print(f"\n[{current}/{total_experiments}] {model_name} - {oracle_name} - repeat {repeat + 1}/{NUM_REPETITIONS} (seed={seed})")

            output_dir = base_dir / model_name / oracle_name / f"repeat_{repeat}"

            try:
                run_single_experiment(
                    model_name=model_name,
                    model_class=model_class,
                    yaml_name=yaml_name,
                    oracle_name=oracle_name,
                    output_dir=output_dir,
                    seed=seed,
                    repeat=repeat,
                )
            except Exception as e:
                print(f"    ERROR: {e}")

    # Always regenerate CSV from all YAML files at the end
    regenerate_csv_from_yaml(base_dir)

    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    rng = np.random.default_rng(DEFAULT_SEED)
    print(f"Running {NUM_REPETITIONS} repetitions per model-oracle combination")
    print(f"Initial random seed: {DEFAULT_SEED}")
    run_all_experiments(rng)
