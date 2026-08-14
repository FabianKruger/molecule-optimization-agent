# Run GraphGA baseline for the IC50 MPro + QED + Novelty experiment.
# Must be run from the mol-opt pixi environment:
#   cd /path/to/mol-opt && pixi run python /path/to/this/script.py
from __future__ import annotations

import json
import math
import sys
import yaml
from pathlib import Path

from molopt.graph_ga import GraphGA  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import MACCSkeys

from molopt_agent.oracles.novel import NovelOracle
from molopt_agent.oracles.qed import ExplainableQedOracle

MAX_ORACLE_CALLS = 50
N_REPLICATES = 3
QED_THRESHOLD = 0.6
RESULTS_DIR = ROOT / "data/results/ic50mpro_qed/graph_ga"


def predict_ic50_nM(smiles: str, model: xgb.XGBRegressor) -> float | None:
    """Predict IC50 in nM from SMILES without SHAP (avoids version conflict)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = np.array(list(MACCSkeys.GenMACCSKeys(mol).ToBitString()), dtype=np.float32)
    pic50 = model.predict(fp.reshape(1, -1))[0]
    return float(10 ** (9 - pic50))


def mol_buffer_to_trace(mol_buffer: dict, eval_cache: dict) -> list[dict]:
    entries = []
    for smiles, (opt_score, call_idx) in mol_buffer.items():
        cache = eval_cache.get(smiles)
        if cache is None:
            continue
        entries.append({
            "iteration": int(call_idx),
            "smiles": smiles,
            "score": cache["ic50"],
            "scores": {"IC50": cache["ic50"], "QED": cache["qed"], "Novelty": cache["novelty"]},
        })
    return sorted(entries, key=lambda x: x["iteration"])


def run(rep: int, seed: int) -> None:
    out_json = RESULTS_DIR / f"ic50mpro_qed_rep{rep}.json"
    if out_json.exists():
        print(f"  Skipping rep{rep} (already exists)")
        return

    tmp_dir = RESULTS_DIR / f"_tmp_rep{rep}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model = xgb.XGBRegressor()
    model.load_model(str(ROOT / "data/xgboost/xgb_maccs_best.json"))
    qed_oracle = ExplainableQedOracle()
    novel_oracle = NovelOracle()
    eval_cache: dict = {}

    def evaluator(smiles: str) -> float:
        ic50 = predict_ic50_nM(smiles, model)
        if ic50 is None:
            return 0.0
        qed = qed_oracle(smiles)["score"]
        novelty = novel_oracle(smiles)["score"]
        eval_cache[smiles] = {"ic50": ic50, "qed": qed, "novelty": novelty}
        # Convert to maximization score: pIC50, penalise low QED or non-novel molecules
        if qed >= QED_THRESHOLD and novelty == 1.0:
            return 9.0 - math.log10(max(ic50, 1e-3))
        return 0.0

    evaluator.__name__ = "ic50mpro_oracle"

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

    yaml_file = next(tmp_dir.glob("results_graph_ga_ic50mpro_oracle_*.yaml"))
    mol_buffer = yaml.safe_load(yaml_file.read_text())
    trace = mol_buffer_to_trace(mol_buffer, eval_cache)

    out_json.write_text(json.dumps({"trace": trace}, indent=2))
    print(f"  Saved {len(trace)} oracle calls → {out_json.name}")

    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for rep in range(1, N_REPLICATES + 1):
        print(f"rep{rep}/{N_REPLICATES}")
        run(rep, seed=rep + 41)
    print("Done.")
