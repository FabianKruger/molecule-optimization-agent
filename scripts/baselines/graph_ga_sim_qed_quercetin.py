# Run GraphGA baseline for the Quercetin similarity+QED experiment.
# Must be run from the mol-opt pixi environment:
#   cd /path/to/mol-opt && pixi run python /path/to/this/script.py
from __future__ import annotations

import json
import sys
import yaml
from pathlib import Path

from molopt.graph_ga import GraphGA  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from molopt_agent.oracles.composite import CompositeOracle
from molopt_agent.oracles.qed import ExplainableQedOracle
from molopt_agent.oracles.similarity import SimilarityOracle

QUERCETIN_SMILES = "O=C1c3c(O/C(=C1/O)c2ccc(O)c(O)c2)cc(O)cc3O"
MAX_ORACLE_CALLS = 1000
N_REPLICATES = 3
RESULTS_DIR = ROOT / "data/results/similarity_qed_quercetin/graph_ga"


def create_oracle() -> CompositeOracle:
    return CompositeOracle(
        oracles=[ExplainableQedOracle(), SimilarityOracle(QUERCETIN_SMILES)],
        weights=[0.5, 0.5],
        names=["QED", "Similarity"],
    )


def mol_buffer_to_trace(mol_buffer: dict) -> list[dict]:
    entries = [
        {"iteration": int(v[1]), "smiles": k, "score": float(v[0])}
        for k, v in mol_buffer.items()
    ]
    return sorted(entries, key=lambda x: x["iteration"])


def run(rep: int, seed: int) -> None:
    out_json = RESULTS_DIR / f"sim_qed_relevant_rep{rep}.json"
    if out_json.exists():
        print(f"  Skipping rep{rep} (already exists)")
        return

    tmp_dir = RESULTS_DIR / f"_tmp_rep{rep}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    oracle = create_oracle()

    def evaluator(smiles: str) -> float:
        return oracle(smiles)["score"]

    evaluator.__name__ = "sim_qed_oracle"

    optimizer = GraphGA(
        smi_file=None,
        n_jobs=1,
        max_oracle_calls=MAX_ORACLE_CALLS,
        freq_log=MAX_ORACLE_CALLS,
        output_dir=str(tmp_dir),
        log_results=False,
    )
    # mol-opt's Oracle uses a mutable default argument `mol_buffer={}` which is shared
    # across all instances. Reset it here so each replicate starts with a clean cache.
    optimizer.oracle.mol_buffer = {}
    optimizer.optimize(evaluator, patience=MAX_ORACLE_CALLS, seed=seed)

    yaml_file = next(tmp_dir.glob("results_graph_ga_sim_qed_oracle_*.yaml"))
    mol_buffer = yaml.safe_load(yaml_file.read_text())
    trace = mol_buffer_to_trace(mol_buffer)

    out_json.write_text(json.dumps({"trace": trace}, indent=2))
    print(f"  Saved {len(trace)} oracle calls → {out_json.name}")

    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for rep in range(1, N_REPLICATES + 1):
        print(f"rep{rep}/{N_REPLICATES}")
        run(rep, seed=rep - 1)
    print("Done.")
