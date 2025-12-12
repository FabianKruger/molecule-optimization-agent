# This script needs to be run with an environment that is build from https://github.com/wenhao-gao/mol-opt
from __future__ import annotations
from typing import Callable
from molopt.reinvent import REINVENT # type: ignore

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from molopt_agent.oracles.qed import ExplainableQedOracle # type: ignore
from molopt_agent.oracles.similarity import SimilarityOracle # type: ignore
from molopt_agent.oracles.composite import CompositeOracle # type: ignore


def create_composite_oracle(target_smiles: str) -> CompositeOracle:
    qed_oracle = ExplainableQedOracle()
    similarity_oracle = SimilarityOracle(target_smiles)
    return CompositeOracle(oracles=[qed_oracle, similarity_oracle], weights=[0.5, 0.5], names=["QED", "Similarity"])

def oracle_wrapper(CompositeOracle: CompositeOracle) -> Callable[str, float]:
    return lambda smiles: CompositeOracle(smiles)['score']

def run(target_smiles: str, results_dir: str, seed: int) -> None:
    optimizer = REINVENT(smi_file=None, n_jobs=-1, max_oracle_calls=1000, freq_log=1, output_dir = results_dir, log_results=True) 
    oracle_function = oracle_wrapper(create_composite_oracle(target_smiles))
    optimizer.optimize(oracle_function, patience=1000, seed=seed)


if __name__ == "__main__":
    directory = ROOT / "data/results/reinvent/quercetin_sim_qed"
    repetitions = 5
    target_smiles = "O=C1c3c(O/C(=C1/O)c2ccc(O)c(O)c2)cc(O)cc3O"

    for repetition in range(repetitions):
        results_dir = directory / f"rep_{repetition}"
        run(target_smiles, str(results_dir), seed=repetition)
        print(f"Processed repetition {repetition}")