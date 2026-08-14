# Run GP-BO baseline for the similarity+QED pubchem experiment.
# Must be run from the mol-opt pixi environment:
#   cd /path/to/mol-opt && pixi run python /path/to/this/script.py
from __future__ import annotations

import json
import sys
import yaml
from pathlib import Path

from molopt.gpbo import GPBO  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from molopt_agent.oracles.composite import CompositeOracle
from molopt_agent.oracles.qed import ExplainableQedOracle
from molopt_agent.oracles.similarity import SimilarityOracle

MAX_ORACLE_CALLS = 50
RESULTS_DIR = ROOT / "data/results/similarity_qed_pubchem/gpbo"
MOLECULES_FILE = ROOT / "data/molecules/sampled_molecules.txt"


def create_oracle(target_smiles: str) -> CompositeOracle:
    return CompositeOracle(
        oracles=[ExplainableQedOracle(), SimilarityOracle(target_smiles)],
        weights=[0.5, 0.5],
        names=["QED", "Similarity"],
    )


def mol_buffer_to_trace(mol_buffer: dict) -> list[dict]:
    """Convert mol-opt mol_buffer {smiles: [score, call_idx]} to ordered trace."""
    entries = [
        {"iteration": int(v[1]), "smiles": k, "score": float(v[0])}
        for k, v in mol_buffer.items()
    ]
    return sorted(entries, key=lambda x: x["iteration"])


def run(mol_idx: int, target_smiles: str) -> None:
    out_json = RESULTS_DIR / f"sim_qed_pubchem_mol{mol_idx}.json"
    if out_json.exists():
        print(f"  Skipping mol{mol_idx} (already exists)")
        return

    tmp_dir = RESULTS_DIR / f"_tmp_mol{mol_idx}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    oracle = create_oracle(target_smiles)

    def evaluator(smiles: str) -> float:
        return oracle(smiles)["score"]

    evaluator.__name__ = "sim_qed_oracle"

    optimizer = GPBO(
        smi_file=None,
        n_jobs=1,
        max_oracle_calls=MAX_ORACLE_CALLS,
        freq_log=MAX_ORACLE_CALLS,
        output_dir=str(tmp_dir),
        log_results=False,
    )
    # mol-opt's Oracle uses a mutable default argument `mol_buffer={}` which is shared
    # across all instances. Reset it here so each run starts with a clean cache.
    optimizer.oracle.mol_buffer = {}
    optimizer.optimize(evaluator, patience=MAX_ORACLE_CALLS, seed=0)

    yaml_file = next(tmp_dir.glob("results_gp_bo_sim_qed_oracle_*.yaml"))
    mol_buffer = yaml.safe_load(yaml_file.read_text())
    trace = mol_buffer_to_trace(mol_buffer)

    out_json.write_text(json.dumps({"trace": trace}, indent=2))
    print(f"  Saved {len(trace)} oracle calls → {out_json.name}")

    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


def load_molecules(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    molecules = load_molecules(MOLECULES_FILE)
    for idx, smiles in enumerate(molecules, start=1):
        print(f"mol{idx}/{len(molecules)}: {smiles[:40]}...")
        run(idx, smiles)
    print("Done.")
