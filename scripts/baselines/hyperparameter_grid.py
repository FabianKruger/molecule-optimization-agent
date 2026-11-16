"""Hyperparameter grid for the classical optimisation baselines.

mol-opt is added to sys.path rather than installed, and the small part of the
TDC API it touches is stubbed out. That keeps the oracle implementation, RDKit
version included, identical to the one SEISMO is scored against, and removes the
TDC download from the reproduction path.

Run with:

    pixi run -e baselines python scripts/baselines/hyperparameter_grid.py \
        --molopt-root ../mol-opt \
        --zinc-pool /path/to/zinc.tab \
        --task sim_qed_pubchem --method all

The ZINC 250k pool is the starting pool mol-opt samples from. It is the same
file TDC's MolGen('ZINC') provides, obtainable from
https://dataverse.harvard.edu/api/access/datafile/4170963
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

QED_THRESHOLD = 0.6
NOVELTY_THRESHOLD = 0.5
# Invalid molecules are floored at pIC50 = 0
PIC50_FLOOR = 0.0


# ---------------------------------------------------------------------------
#  mol-opt needs a few TDC entry points that are irrelevant to optimisation
# ---------------------------------------------------------------------------
def install_tdc_stub(zinc_pool: Path) -> None:
    """Provide the TDC surface mol-opt touches, without installing pytdc.

    mol-opt constructs an SA scorer, a diversity evaluator and a structural
    filter in its Oracle and BaseOptimizer, and reads ZINC as the starting pool.
    The first three are only consumed by mol-opt's own logging, which this script
    does not use, so they return constants. The starting pool is read from a
    local copy of the same ZINC file TDC would download.
    """
    import pandas as pd

    pool = read_smiles_pool(zinc_pool)
    print(f"starting pool: {len(pool)} molecules from {zinc_pool.name}")

    class _StubOracle:
        """Stands in for tdc.Oracle(name='SA'), used only in logging."""

        def __init__(self, name=None, *args, **kwargs):
            self.name = name or "stub"

        def __call__(self, smiles):
            if isinstance(smiles, (list, tuple)):
                return [0.0] * len(smiles)
            return 0.0

    class _StubEvaluator:
        """Stands in for tdc.Evaluator(name='Diversity'), used only in logging."""

        def __init__(self, name=None, *args, **kwargs):
            self.name = name or "stub"

        def __call__(self, smiles):
            return 0.0

    class _StubMolFilter:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, smiles):
            return smiles

    class _StubMolGen:
        def __init__(self, name=None, *args, **kwargs):
            self.name = name or "ZINC"

        def get_data(self):
            return pd.DataFrame({"smiles": pool})

    tdc = types.ModuleType("tdc")
    tdc.Oracle = _StubOracle
    tdc.Evaluator = _StubEvaluator

    generation = types.ModuleType("tdc.generation")
    generation.MolGen = _StubMolGen

    chem_utils = types.ModuleType("tdc.chem_utils")
    oracle_mod = types.ModuleType("tdc.chem_utils.oracle")
    filter_mod = types.ModuleType("tdc.chem_utils.oracle.filter")
    filter_mod.MolFilter = _StubMolFilter
    oracle_mod.filter = filter_mod
    chem_utils.oracle = oracle_mod
    tdc.chem_utils = chem_utils
    tdc.generation = generation

    for name, module in [
        ("tdc", tdc),
        ("tdc.generation", generation),
        ("tdc.chem_utils", chem_utils),
        ("tdc.chem_utils.oracle", oracle_mod),
        ("tdc.chem_utils.oracle.filter", filter_mod),
    ]:
        sys.modules[name] = module


def install_gpbo_compat() -> None:
    """Two compatibility shims GP-BO needs on a current scientific stack.

    1. mol-opt calls botorch.optim.fit.fit_gpytorch_scipy, renamed in botorch
       0.9. Both call sites discard the return value and only need the marginal
       log likelihood fitted in place, which fit_gpytorch_mll does with the same
       scipy L-BFGS-B optimiser.
    2. mol-opt passes a set to random.sample, which Python 3.11 no longer
       accepts. The set is the starting population pool, at most
       initial_population_size molecules. Sorting it first keeps the draw
       deterministic under the run seed, which list(set) would not because string
       hashing is randomised per process.

    GP-BO's upstream configuration never reaches the GP fitting path at a 50 call
    budget, since its 340 molecule initial sample already exhausts the budget, so
    only the budget-appropriate configurations are affected by the first shim.
    """
    import botorch.fit
    import botorch.optim.fit

    if not hasattr(botorch.optim.fit, "fit_gpytorch_scipy"):

        def fit_gpytorch_scipy(mll, *args, **kwargs):
            return botorch.fit.fit_gpytorch_mll(mll)

        botorch.optim.fit.fit_gpytorch_scipy = fit_gpytorch_scipy
        print("shim: botorch.optim.fit.fit_gpytorch_scipy -> fit_gpytorch_mll")

    if not getattr(random.sample, "_accepts_sets", False):
        original_sample = random.sample

        def sample(population, k, *args, **kwargs):
            if isinstance(population, (set, frozenset, dict)):
                population = sorted(population)
            return original_sample(population, k, *args, **kwargs)

        sample._accepts_sets = True
        random.sample = sample
        print("shim: random.sample accepts sets again, ordered deterministically")


def read_smiles_pool(path: Path) -> list[str]:
    """Read a ZINC style file, tolerating a header and quoted SMILES."""
    smiles = []
    with open(path, newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            value = row[0].strip().strip('"')
            if not value or value.lower() == "smiles":
                continue
            smiles.append(value)
    if not smiles:
        raise ValueError(f"no SMILES found in {path}")
    return smiles


# ---------------------------------------------------------------------------
#  Novelty with a disk cache, to keep PubChem traffic civil
# ---------------------------------------------------------------------------
class CachedNovelOracle:
    """NovelOracle with a persistent cache and a request rate limit.

    One PubChem REST lookup per unique molecule, reused across configurations
    and replicates. PubChem asks for no more than five requests per second.
    """

    def __init__(self, cache_path: Path, min_interval: float = 0.25):
        from molopt_agent.oracles.novel import NovelOracle

        self._oracle = NovelOracle()
        self._cache_path = cache_path
        self._min_interval = min_interval
        self._last_request = 0.0
        self._dirty = 0
        if cache_path.exists():
            self._cache = json.loads(cache_path.read_text())
        else:
            self._cache = {}
        self.live_lookups = 0

    def __call__(self, smiles: str) -> float:
        if smiles in self._cache:
            return self._cache[smiles]

        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        score = float(self._oracle(smiles)["score"])
        self._last_request = time.monotonic()
        self.live_lookups += 1

        self._cache[smiles] = score
        self._dirty += 1
        if self._dirty >= 25:
            self.flush()
        return score

    def flush(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache))
        self._dirty = 0


# ---------------------------------------------------------------------------
#  Tasks
# ---------------------------------------------------------------------------
def make_ic50_evaluator(novelty: CachedNovelOracle):
    """IC50-QED-Novel: maximise pIC50, floored at 0 when a constraint fails."""
    import xgboost as xgb
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys

    from molopt_agent.oracles.qed import ExplainableQedOracle

    model = xgb.XGBRegressor()
    model.load_model(str(ROOT / "data/xgboost/xgb_maccs_best.json"))
    qed_oracle = ExplainableQedOracle()
    details: dict[str, dict] = {}

    def evaluator(smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        fp = np.array(list(MACCSkeys.GenMACCSKeys(mol).ToBitString()), dtype=np.float32)
        pic50 = float(model.predict(fp.reshape(1, -1))[0])
        ic50_nM = 10 ** (9 - pic50)
        qed = float(qed_oracle(smiles)["score"])
        novel = novelty(smiles)
        valid = qed >= QED_THRESHOLD and novel >= NOVELTY_THRESHOLD
        score = max(pic50, PIC50_FLOOR) if valid else PIC50_FLOOR
        details[smiles] = {"ic50_nM": ic50_nM, "qed": qed, "novelty": novel, "pic50": pic50}
        return score

    evaluator.__name__ = "ic50mpro_oracle"
    return evaluator, details


def make_sim_qed_evaluator(target_smiles: str):
    """Sim-QED: maximise the mean of Tanimoto similarity and QED."""
    from molopt_agent.oracles.composite import CompositeOracle
    from molopt_agent.oracles.qed import ExplainableQedOracle
    from molopt_agent.oracles.similarity import SimilarityOracle

    oracle = CompositeOracle(
        oracles=[ExplainableQedOracle(), SimilarityOracle(target_smiles)],
        weights=[0.5, 0.5],
        names=["QED", "Similarity"],
    )
    details: dict[str, dict] = {}

    def evaluator(smiles: str) -> float:
        result = oracle(smiles)
        details[smiles] = {"score": float(result["score"])}
        return float(result["score"])

    evaluator.__name__ = "sim_qed_oracle"
    return evaluator, details


# ---------------------------------------------------------------------------
#  Metric, matching analysis/paper/xai_vs_no_xai_ic50mpro_extended.ipynb
# ---------------------------------------------------------------------------
def top1_auc(trace: list[dict], budget: int, floor: float = 0.0) -> float:
    """Mean best-so-far score over the budget, by the trapezoidal rule.

    The best-so-far curve is integrated trapezoidally and divided by the budget,
    which is what PMO's own top_auc does.
    """
    best_at_call: dict[int, float] = {}
    best = floor
    for entry in sorted(trace, key=lambda e: e["iteration"]):
        call = int(entry["iteration"])
        if call > budget:
            break
        best = max(best, float(entry["score"]))
        best_at_call[call] = best

    if not best_at_call:
        return floor

    auc = 0.0
    prev_call, prev_best = 0, floor
    for call in sorted(best_at_call):
        auc += (call - prev_call) * (best_at_call[call] + prev_best) / 2
        prev_call, prev_best = call, best_at_call[call]
    auc += prev_best * (budget - prev_call)
    return auc / budget



def buffer_to_trace(mol_buffer: dict, details: dict) -> list[dict]:
    trace = []
    for smiles, (score, call_idx) in mol_buffer.items():
        entry = {"iteration": int(call_idx), "smiles": smiles, "score": float(score)}
        entry.update(details.get(smiles, {}))
        trace.append(entry)
    return sorted(trace, key=lambda e: e["iteration"])


# ---------------------------------------------------------------------------
#  Grids. The first entry of each is the upstream default used in the
#  submitted manuscript.
# ---------------------------------------------------------------------------
GRIDS: dict[str, list[dict]] = {
    "graph_ga": [
        {"label": "default", "population_size": 120, "offspring_size": 70, "mutation_rate": 0.067},
        {"label": "pop 40 / off 20", "population_size": 40, "offspring_size": 20, "mutation_rate": 0.067},
        {"label": "pop 20 / off 10", "population_size": 20, "offspring_size": 10, "mutation_rate": 0.067},
        {"label": "pop 10 / off 5", "population_size": 10, "offspring_size": 5, "mutation_rate": 0.067},
        {"label": "pop 20 / off 10, mut 0.2", "population_size": 20, "offspring_size": 10, "mutation_rate": 0.2},
    ],
    "gpbo": [
        {"label": "default", "initial_population_size": 340, "bo_batch_size": 1180, "max_bo_iter": 10000},
        {"label": "init 20 / batch 1", "initial_population_size": 20, "bo_batch_size": 1, "max_bo_iter": 100},
        {"label": "init 10 / batch 5", "initial_population_size": 10, "bo_batch_size": 5, "max_bo_iter": 100},
    ],
    "reinvent": [
        {"label": "default", "batch_size": 64, "sigma": 500, "experience_replay": 24},
        {"label": "batch 16", "batch_size": 16, "sigma": 500, "experience_replay": 24},
        {"label": "batch 8", "batch_size": 8, "sigma": 500, "experience_replay": 24},
        {"label": "batch 4", "batch_size": 4, "sigma": 500, "experience_replay": 24},
        {"label": "batch 8, replay 8", "batch_size": 8, "sigma": 500, "experience_replay": 8},
    ],
    "hill_climb": [
        {"label": "mutation only", "mutation_attempts": 20},
    ],
}

# Oracle calls consumed before the method can take its first update step.
FIRST_UPDATE_AFTER = {
    "graph_ga": lambda cfg: cfg["population_size"],
    "gpbo": lambda cfg: cfg["initial_population_size"],
    "reinvent": lambda cfg: cfg["batch_size"],
    "hill_climb": lambda cfg: 1,
}


# ---------------------------------------------------------------------------
#  Runners
# ---------------------------------------------------------------------------
def run_molopt_method(method: str, params: dict, evaluator, details, pool, budget, seed, out_dir):
    """Run one mol-opt optimiser with hyperparameters passed as kwargs."""
    # Imported per method so a broken dependency in one optimiser, for instance
    # a gpytorch API change in GP-BO, cannot block the others.
    if method == "graph_ga":
        from molopt.graph_ga import GraphGA as optimizer_cls
    elif method == "gpbo":
        install_gpbo_compat()
        from molopt.gpbo import GPBO as optimizer_cls
    elif method == "reinvent":
        from molopt.reinvent import REINVENT as optimizer_cls
    else:
        raise ValueError(f"unknown mol-opt method {method}")

    kwargs = {k: v for k, v in params.items() if k != "label"}

    optimizer = optimizer_cls(
        smi_file=None,
        n_jobs=1,
        max_oracle_calls=budget,
        freq_log=budget,
        output_dir=str(out_dir),
        log_results=False,
    )
    # mol-opt's Oracle has a mutable default argument shared across instances.
    optimizer.oracle.mol_buffer = {}
    optimizer.optimize(evaluator, patience=budget, seed=seed, **kwargs)

    # optimize() ends in reset(), which discards the oracle and its buffer, so
    # the trace is read from the YAML that save_result wrote beforehand.
    import yaml

    dumps = sorted(out_dir.glob("results_*.yaml"))
    if not dumps:
        raise RuntimeError(f"{method} wrote no results YAML to {out_dir}")
    mol_buffer = yaml.safe_load(dumps[-1].read_text())
    return buffer_to_trace(mol_buffer, details)


def run_hill_climb(params, evaluator, details, pool, budget, seed):
    """Sequential local search: mutate the incumbent, accept on improvement.

    One oracle call per proposal, so the incumbent updates after every evaluated
    molecule. Repeat proposals are skipped without consuming budget, matching
    mol-opt's deduplicating oracle.
    """
    from rdkit import Chem

    from molopt.graph_ga.mutate import mutate

    rng = random.Random(seed)
    np.random.seed(seed)
    random.seed(seed)

    attempts = params.get("mutation_attempts", 20)
    evaluated: dict[str, float] = {}
    trace: list[dict] = []

    def evaluate(smiles: str) -> float | None:
        if smiles in evaluated:
            return None  # no budget consumed for a repeat
        score = float(evaluator(smiles))
        evaluated[smiles] = score
        entry = {"iteration": len(evaluated), "smiles": smiles, "score": score}
        entry.update(details.get(smiles, {}))
        trace.append(entry)
        return score

    incumbent, incumbent_score = None, -math.inf
    while len(evaluated) < budget:
        if incumbent is None:
            candidate_smiles = rng.choice(pool)
        else:
            candidate_smiles = None
            for _ in range(attempts):
                child = mutate(Chem.Mol(incumbent), 1.0)
                if child is None:
                    continue
                try:
                    smiles = Chem.MolToSmiles(child)
                except Exception:
                    continue
                if smiles and smiles not in evaluated:
                    candidate_smiles = smiles
                    break
            if candidate_smiles is None:  # local search stuck, restart
                candidate_smiles = rng.choice(pool)

        score = evaluate(candidate_smiles)
        if score is None:
            continue
        if score > incumbent_score:
            mol = Chem.MolFromSmiles(candidate_smiles)
            if mol is not None:
                incumbent, incumbent_score = mol, score

    return trace


def summarise(task_dir: Path, budget: int) -> list[dict]:
    """Rebuild the full table from every run saved on disk.

    Each method is run in its own invocation, so the summary is reconstructed
    across all of them rather than only the ones in the current call.
    """
    rows = []
    for method, configs in GRIDS.items():
        for params in configs:
            label = params["label"]
            slug = config_slug(label)
            run_files = sorted((task_dir / method / slug).glob("*.json"))
            if not run_files:
                continue
            aucs, bests = [], []
            for path in run_files:
                trace = json.loads(path.read_text())["trace"]
                aucs.append(top1_auc(trace, budget))
                bests.append(max([e["score"] for e in trace], default=0.0))
            rows.append(
                {
                    "method": method,
                    "label": label,
                    "config": json.dumps({k: v for k, v in params.items() if k != "label"}),
                    "calls_before_first_update": FIRST_UPDATE_AFTER[method](params),
                    "n_runs": len(aucs),
                    "auc_mean": float(np.mean(aucs)),
                    "auc_std": float(np.std(aucs)),
                    "best_mean": float(np.mean(bests)),
                }
            )
    return rows


def config_slug(label: str) -> str:
    return label.replace(" ", "_").replace("/", "").replace(",", "")


def run_one(method, params, task, target, budget, seed, novelty, pool, out_dir):
    if task == "ic50":
        evaluator, details = make_ic50_evaluator(novelty)
    else:
        evaluator, details = make_sim_qed_evaluator(target)

    if method == "hill_climb":
        trace = run_hill_climb(params, evaluator, details, pool, budget, seed)
    else:
        trace = run_molopt_method(
            method, params, evaluator, details, pool, budget, seed, out_dir
        )
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molopt-root", type=Path, required=True)
    parser.add_argument("--zinc-pool", type=Path, required=True)
    parser.add_argument("--task", choices=["ic50", "sim_qed_pubchem"], required=True)
    parser.add_argument("--method", default="all", choices=["all", *GRIDS])
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--reps", type=int, default=5, help="replicates for the ic50 task")
    parser.add_argument("--targets", type=int, default=20, help="targets for sim_qed_pubchem")
    parser.add_argument("--out", type=Path, default=ROOT / "data/results/baseline_grid")
    parser.add_argument(
        "--novelty-cache", type=Path, default=ROOT / "data/results/baseline_grid/novelty_cache.json"
    )
    parser.add_argument(
        "--pubchem-min-interval",
        type=float,
        default=0.25,
        help=(
            "seconds between live PubChem lookups. The default of 0.25 is the "
            "four requests per second PubChem permits, and it is enforced PER "
            "PROCESS, so running N copies of this script on the ic50 task at "
            "once means 4N requests per second. Raise it to 0.25*N in that case. "
            "The sim_qed_pubchem task makes no live lookups at all, so running "
            "that task in parallel needs no change."
        ),
    )
    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "comma-separated configuration labels to run, as they appear in GRIDS. "
            "Default is every configuration for the selected method."
        ),
    )
    parser.add_argument(
        "--run-indices",
        default=None,
        help=(
            "comma-separated 1-based run indices to execute (ic50 replicates or "
            "sim_qed_pubchem targets). This splits one grid across several "
            "processes running at the same time, each taking a different subset, "
            "which is worth doing because a single run is single-threaded. Each "
            "run writes its own result file and its own scratch directory, so "
            "the processes do not interfere as long as no index is handed to two "
            "of them. Splitting the ic50 task this way also needs "
            "--pubchem-min-interval and --novelty-cache set per process."
        ),
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help=(
            "do not rebuild the summary CSV. Set this when running several "
            "processes at once so they do not write the same file on top of each "
            "other, then make one final call without it to write the summary."
        ),
    )
    args = parser.parse_args()

    molopt_root = args.molopt_root.resolve()
    if not (molopt_root / "molopt").is_dir():
        raise SystemExit(f"no molopt package under {molopt_root}")
    sys.path.insert(0, str(molopt_root))

    install_tdc_stub(args.zinc_pool.resolve())
    pool = read_smiles_pool(args.zinc_pool.resolve())

    novelty = None
    if args.task == "ic50":
        novelty = CachedNovelOracle(args.novelty_cache, args.pubchem_min_interval)

    if args.task == "ic50":
        runs = [(f"rep{i}", None, 41 + i) for i in range(1, args.reps + 1)]
    else:
        targets = [
            line.strip()
            for line in (ROOT / "data/molecules/sampled_molecules.txt").read_text().splitlines()
            if line.strip()
        ][: args.targets]
        runs = [(f"mol{i}", smiles, 41 + i) for i, smiles in enumerate(targets, start=1)]

    if args.run_indices:
        wanted = {int(i) for i in args.run_indices.split(",")}
        unknown = wanted - set(range(1, len(runs) + 1))
        if unknown:
            raise SystemExit(f"run indices out of range 1..{len(runs)}: {sorted(unknown)}")
        runs = [run for i, run in enumerate(runs, start=1) if i in wanted]

    methods = list(GRIDS) if args.method == "all" else [args.method]
    task_dir = args.out / args.task

    wanted_labels = None
    if args.labels:
        wanted_labels = [label.strip() for label in args.labels.split(",")]
        known = {params["label"] for method in methods for params in GRIDS[method]}
        unknown = [label for label in wanted_labels if label not in known]
        if unknown:
            raise SystemExit(
                f"unknown configuration labels {unknown}; known labels are {sorted(known)}"
            )

    for method in methods:
        for params in GRIDS[method]:
            label = params["label"]
            if wanted_labels is not None and label not in wanted_labels:
                continue
            slug = config_slug(label)
            aucs = []
            for run_name, target, seed in runs:
                out_json = task_dir / method / slug / f"{run_name}.json"
                if out_json.exists():
                    trace = json.loads(out_json.read_text())["trace"]
                else:
                    tmp_dir = task_dir / method / slug / f"_tmp_{run_name}"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    trace = run_one(
                        method, params, args.task, target, args.budget, seed, novelty, pool, tmp_dir
                    )
                    out_json.parent.mkdir(parents=True, exist_ok=True)
                    out_json.write_text(
                        json.dumps(
                            {
                                "method": method,
                                "config": {k: v for k, v in params.items() if k != "label"},
                                "label": label,
                                "task": args.task,
                                "target": target,
                                "seed": seed,
                                "budget": args.budget,
                                "trace": trace,
                            },
                            indent=2,
                        )
                    )
                    for leftover in tmp_dir.glob("*"):
                        leftover.unlink()
                    tmp_dir.rmdir()
                    if novelty is not None:
                        novelty.flush()

                auc = top1_auc(trace, args.budget)
                aucs.append(auc)
                print(f"  {method:11s} {label:24s} {run_name:6s} AUC {auc:.3f}")

    if args.skip_summary:
        print("\nruns complete, summary skipped")
        if novelty is not None:
            novelty.flush()
            print(f"live PubChem lookups this run: {novelty.live_lookups}")
        return

    rows = summarise(task_dir, args.budget)

    if novelty is not None:
        novelty.flush()
        print(f"\nlive PubChem lookups this run: {novelty.live_lookups}")

    summary_csv = args.out / f"{args.task}_grid_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\n{'method':12s} {'configuration':26s} {'calls to 1st update':>20s} "
        f"{'n':>4s} {'AUC':>16s}"
    )
    for row in rows:
        print(
            f"{row['method']:12s} {row['label']:26s} {row['calls_before_first_update']:>20d} "
            f"{row['n_runs']:>4d} {row['auc_mean']:>8.3f} +/- {row['auc_std']:.3f}"
        )
    print(f"\nwrote {summary_csv}")


if __name__ == "__main__":
    main()
