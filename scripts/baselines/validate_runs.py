"""Integrity check for saved baseline runs.

A run file existing does not mean the run is sound: an oracle exception or a
killed process can leave a short trace behind, and a short trace silently
depresses the AUC it contributes to. This checks every saved run for the
properties the appendix table assumes.

    pixi run -e baselines python scripts/baselines/validate_runs.py
    pixi run -e baselines python scripts/baselines/validate_runs.py \
        --out data/results/iso_quality

Both result layouts are understood: the hyperparameter grid, which stores runs
under <task>/<method>/<config slug>/<run>.json, and the iso-quality experiment,
which stores them under <task>/<method>/<run>.json because it only ever runs the
upstream default. Each run's budget is read from its own file, so a tree holding
several horizons validates correctly.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hyperparameter_grid import GRIDS, config_slug  # noqa: E402


def expected_seed(run_name: str) -> int | None:
    """Seed implied by a run's file name, or None if the name does not imply one.

    Grid runs are rep<i> or mol<i> and use seed 41+i. Iso-quality runs are
    mol<i>_rep<r> and use 41+i+100*(r-1), so that replicate 1 reuses the grid's
    seed and its first 50 calls can be compared against it.
    """
    if "_rep" in run_name:
        head, _, rep = run_name.partition("_rep")
        if head.startswith("mol"):
            try:
                return 41 + int(head[3:]) + 100 * (int(rep) - 1)
            except ValueError:
                return None
        return None
    for prefix in ("rep", "mol"):
        if run_name.startswith(prefix):
            try:
                return 41 + int(run_name[len(prefix):])
            except ValueError:
                return None
    return None


def run_name(path: Path) -> str:
    """File name without extensions. Path.stem leaves the .json on a .json.gz."""
    return path.name.split(".")[0]


def load_run(path: Path) -> dict:
    """Read a run file, plain or gzipped. Cluster runs arrive gzipped."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def check(path: Path, budget: int) -> list[str]:
    problems = []
    try:
        data = load_run(path)
    except json.JSONDecodeError as error:
        return [f"unreadable JSON ({error})"]

    # Each run records the budget it was given. Trusting that rather than one
    # global value lets a tree mixing horizons validate correctly, which the
    # iso-quality results do.
    budget = int(data.get("budget") or budget)

    trace = data.get("trace", [])
    iters = [int(e["iteration"]) for e in trace]

    # mol-opt's budget gating overshoots by at most one call: a batch is scored
    # and only then is the budget found to be spent. top1_auc stops at
    # iteration > budget, so the extra molecule never enters the AUC and confers
    # no advantage. A trace SHORTER than the budget is the real fault, because it
    # silently depresses the AUC it contributes to.
    if len(trace) < budget:
        problems.append(f"only {len(trace)} entries, short of the {budget} call budget")
    elif len(trace) > budget + 1:
        problems.append(
            f"{len(trace)} entries, more than the one-call overshoot {budget + 1} allows"
        )
    if iters and sorted(iters) != list(range(1, len(iters) + 1)):
        dupes = [i for i, c in Counter(iters).items() if c > 1]
        problems.append(
            f"iterations not 1..{len(iters)} (min {min(iters)}, max {max(iters)}"
            + (f", duplicated {sorted(dupes)}" if dupes else "")
            + ")"
        )
    if any("score" not in e for e in trace):
        problems.append("entry without a score")
    if any(e.get("score") is None for e in trace):
        problems.append("entry with a null score")

    name = run_name(path)
    want_seed = expected_seed(name)
    if want_seed is None:
        # A name that implies no seed means the seed goes unchecked, so say so
        # rather than passing the run quietly.
        problems.append(f"file name {name!r} does not imply a seed, so it cannot be checked")
    elif data.get("seed") != want_seed:
        problems.append(f"seed {data.get('seed')}, expected {want_seed} for {name}")

    label = data.get("label")
    method = data.get("method")
    if path.parent.name == method:
        # Iso-quality layout: the run sits directly under its method directory
        # because only the upstream default is ever run.
        if label != "default":
            problems.append(f"label {label!r} under a method directory, expected 'default'")
    elif label is not None and config_slug(label) != path.parent.name:
        problems.append(f"label {label!r} does not match directory {path.parent.name!r}")
    if method is not None and method not in GRIDS:
        problems.append(f"method {method!r} is not in GRIDS")

    known = {p["label"] for params in GRIDS.values() for p in params}
    if label not in known:
        problems.append(f"label {label!r} is not in GRIDS")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/results/baseline_grid",
    )
    parser.add_argument("--budget", type=int, default=50)
    args = parser.parse_args()


    skip = {"novelty_cache.json"}
    files = sorted(
        p
        for p in [*args.out.rglob("*.json"), *args.out.rglob("*.json.gz")]
        if p.name not in skip
    )
    # Checking nothing and reporting no problems is a false pass, so say so.
    if not files:
        raise SystemExit(f"no run files found under {args.out}")

    bad = 0
    for path in files:
        problems = check(path, args.budget)
        if problems:
            bad += 1
            rel = path.relative_to(args.out)
            for problem in problems:
                print(f"FAIL {rel}: {problem}")

    print(f"\nchecked {len(files)} run files, {bad} with problems")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
