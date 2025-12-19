# This script needs to be run with an environment that is build from https://github.com/wenhao-gao/mol-opt
from __future__ import annotations
import os
from typing import Callable
from molopt.reinvent import REINVENT # type: ignore

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from molopt_agent.oracles.qed import ExplainableQedOracle
from molopt_agent.oracles.similarity import SimilarityOracle
from molopt_agent.oracles.composite import CompositeOracle


def create_composite_oracle(target_smiles: str) -> CompositeOracle:
    qed_oracle = ExplainableQedOracle()
    similarity_oracle = SimilarityOracle(target_smiles)
    return CompositeOracle(oracles=[qed_oracle, similarity_oracle], weights=[0.5, 0.5], names=["QED", "Similarity"])

def oracle_wrapper(CompositeOracle: CompositeOracle) -> Callable[str, float]:
    return lambda smiles: CompositeOracle(smiles)['score']

def run(target_smiles: str, results_dir: str) -> None:
    optimizer = REINVENT(smi_file=None, n_jobs=-1, max_oracle_calls=1000, freq_log=1, output_dir = results_dir, log_results=True) 
    oracle_function = oracle_wrapper(create_composite_oracle(target_smiles))
    optimizer.optimize(oracle_function, patience=1000, seed=0)

def load_molecules(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]


if __name__ == "__main__":
    directory = ROOT / "data/results/reinvent/random_molecules_sim_qed"
    molecules_file = ROOT / "data/molecules/sampled_molecules.txt"
    molecules = load_molecules(molecules_file)
    mol = 1
    for molecule in molecules:
        results_dir = directory / f"mol_{mol}"
        run(molecule, str(results_dir))
        mol += 1
        print(f"Processed molecule {mol}")