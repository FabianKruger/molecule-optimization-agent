"""Long-horizon baselines and the oracle-call cost of matching SEISMO.

For every baseline run, how many oracle calls it needs before its best-so-far
molecule matches the best molecule SEISMO found within 50 calls, and the oracle
cost at which paying for LLM calls instead becomes the cheaper route. The LLM cost
of the zero-shot alternative is a separate question, answered by
zero-shot-generation.py --estimate-cost.

Run with:

    pixi run -e baselines python scripts/baselines/iso_quality.py \
        --molopt-root ../mol-opt \
        --zinc-pool /path/to/zinc.tab \
        --budget 10000 --reps 3

A single run is single-threaded, so the 180 runs split cleanly across processes
with --run-indices, each process taking a different subset of targets:

    for i in 1 2 3 4 5; do
        pixi run -e baselines python scripts/baselines/iso_quality.py \
            --molopt-root ../mol-opt --zinc-pool /path/to/zinc.tab \
            --run-indices $(seq -s, $(( (i-1)*4+1 )) $(( i*4 ))) \
            --skip-analysis &
    done
    wait
    pixi run -e baselines python scripts/baselines/iso_quality.py \
        --molopt-root ../mol-opt --zinc-pool /path/to/zinc.tab --analysis-only
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hyperparameter_grid import (  # noqa: E402
    GRIDS,
    ROOT,
    install_tdc_stub,
    read_smiles_pool,
    run_one,
)

SEISMO_DIR = ROOT / "data/results/similarity_qed_pubchem/claude-opus-4.5_full_xai"
SEISMO_BUDGET = 50

SEISMO_LLM_COST_USD = 8.46
SEISMO_LLM_SECONDS_PER_ITER = 6.7

METHODS = ["graph_ga", "reinvent", "gpbo"]
PRETTY = {"graph_ga": "Graph-GA", "reinvent": "REINVENT", "gpbo": "GP BO"}


def default_config(method: str) -> dict:
    """The upstream default, which is the first entry of each grid."""
    params = GRIDS[method][0]
    if params["label"] != "default":
        raise RuntimeError(f"first {method} grid entry is {params['label']!r}, not the default")
    return params


def run_seed(target_index: int, rep: int) -> int:
    return 41 + target_index + 100 * (rep - 1)


# ---------------------------------------------------------------------------
#  Running
# ---------------------------------------------------------------------------
def build_runs(targets: list[str], reps: int) -> list[tuple[str, str, int, int, int]]:
    """One entry per (target, replicate), as (name, smiles, index, rep, seed)."""
    runs = []
    for index, smiles in enumerate(targets, start=1):
        for rep in range(1, reps + 1):
            runs.append((f"mol{index}_rep{rep}", smiles, index, rep, run_seed(index, rep)))
    return runs


def execute(args, pool) -> None:
    targets = [
        line.strip()
        for line in (ROOT / "data/molecules/sampled_molecules.txt").read_text().splitlines()
        if line.strip()
    ][: args.targets]

    if args.run_indices:
        wanted = {int(i) for i in args.run_indices.split(",")}
        unknown = wanted - set(range(1, len(targets) + 1))
        if unknown:
            raise SystemExit(f"target indices out of range 1..{len(targets)}: {sorted(unknown)}")
        targets = [t if i in wanted else None for i, t in enumerate(targets, start=1)]

    # One run cannot be split, since each oracle call depends on the previous
    # one, so a (target, replicate) pair is the smallest unit of work and the
    # right granularity for a job array.
    reps_wanted = set(range(1, args.reps + 1))
    if args.rep_indices:
        reps_wanted = {int(i) for i in args.rep_indices.split(",")}
        unknown = reps_wanted - set(range(1, args.reps + 1))
        if unknown:
            raise SystemExit(f"replicate indices out of range 1..{args.reps}: {sorted(unknown)}")

    task_dir = args.out / "sim_qed_pubchem"

    for method in args.methods:
        params = default_config(method)
        for name, smiles, index, rep, seed in build_runs(targets, args.reps):
            if smiles is None or rep not in reps_wanted:
                continue
            out_json = task_dir / method / f"{name}.json"
            if out_json.exists():
                print(f"  {method:9s} {name:12s} already done")
                continue

            tmp_dir = task_dir / method / f"_tmp_{name}"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            trace = run_one(
                method,
                params,
                "sim_qed_pubchem",
                smiles,
                args.budget,
                seed,
                None,
                pool,
                tmp_dir,
            )
            out_json.write_text(
                json.dumps(
                    {
                        "method": method,
                        "config": {k: v for k, v in params.items() if k != "label"},
                        "label": params["label"],
                        "task": "sim_qed_pubchem",
                        "target": smiles,
                        "target_index": index,
                        "seed": seed,
                        "budget": args.budget,
                        # The baselines' own computation is part of the cost
                        # comparison, so each run records what it took rather
                        # than leaving it to be reconstructed from file times.
                        # Single-threaded, so this is also its core time.
                        "seconds": round(time.perf_counter() - started, 2),
                        "machine": platform.machine(),
                        "trace": trace,
                    },
                    indent=2,
                )
            )
            for leftover in tmp_dir.glob("*"):
                leftover.unlink()
            tmp_dir.rmdir()

            best = max((e["score"] for e in trace), default=0.0)
            elapsed = time.perf_counter() - started
            print(
                f"  {method:9s} {name:12s} best {best:.4f} over {args.budget} calls "
                f"in {elapsed:.1f}s"
            )


# ---------------------------------------------------------------------------
#  Analysis
# ---------------------------------------------------------------------------
def load_run(path: Path) -> dict:
    """Read a run file, plain or gzipped.

    Long-horizon traces are large, so runs produced on the cluster are gzipped
    before being pushed. A 10,000 call trace is 1.3 MB plain and 0.22 MB gzipped.
    """
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def run_files(directory: Path, pattern: str) -> list[Path]:
    """Matching run files, preferring a plain file over a gzipped duplicate."""
    found: dict[str, Path] = {}
    for path in sorted(directory.glob(f"{pattern}.json.gz")):
        found[path.name[: -len(".json.gz")]] = path
    for path in sorted(directory.glob(f"{pattern}.json")):
        found[path.stem] = path
    return [found[name] for name in sorted(found)]


def seismo_best_per_target(targets: int) -> dict[int, float]:
    """Best score SEISMO reached within its 50 call budget, per target."""
    bests = {}
    for index in range(1, targets + 1):
        matches = sorted(SEISMO_DIR.glob(f"*mol{index}_full*.json"))
        if not matches:
            raise SystemExit(f"no SEISMO run file for target mol{index} under {SEISMO_DIR}")
        trace = json.loads(matches[0].read_text())["trace"]
        scores = [
            float(e["score"]) for e in trace if int(e["iteration"]) <= SEISMO_BUDGET
        ]
        if not scores:
            raise SystemExit(f"SEISMO run for mol{index} has no scored iterations")
        bests[index] = max(scores)
    return bests


def calls_to_match(trace: list[dict], threshold: float, budget: int) -> int | None:
    """First oracle call at which best-so-far reaches the threshold.

    mol-opt scores a whole batch before noticing the budget is spent, so traces
    carry one call past the budget. That overshoot is discarded, as it is in the
    AUC definition.
    """
    best = 0.0
    for entry in sorted(trace, key=lambda e: int(e["iteration"])):
        call = int(entry["iteration"])
        if call > budget:
            break
        best = max(best, float(entry["score"]))
        if best >= threshold:
            return call
    return None


def per_run_rows(task_dir: Path, methods: list[str], thresholds: dict[int, float]) -> list[dict]:
    """One row per run: how many calls it needed to match SEISMO, or none."""
    rows = []
    for method in methods:
        for path in run_files(task_dir / method, "mol*_rep*"):
            run = load_run(path)
            index = run["target_index"]
            if index not in thresholds:
                continue
            trace = run["trace"]
            k = calls_to_match(trace, thresholds[index], run["budget"])
            rows.append(
                {
                    "method": method,
                    "run": path.name.split(".")[0],
                    "target_index": index,
                    "seed": run["seed"],
                    "budget": run["budget"],
                    "seismo_best_at_50": round(thresholds[index], 6),
                    "baseline_best_at_budget": round(
                        max((e["score"] for e in trace), default=0.0), 6
                    ),
                    "calls_to_match": k if k is not None else "",
                    "matched": int(k is not None),
                    # Absent in runs produced before timings were recorded.
                    "seconds": run.get("seconds", ""),
                    "machine": run.get("machine", ""),
                }
            )
    return rows


def breakeven(k: float | None, llm_total: float) -> float | str:
    """Oracle cost above which paying for LLM calls to save k-50 calls wins.

    k is the matched-only median, not the censored-aware one. It is the smaller
    of the two, so it yields the higher threshold and therefore the conservative
    claim: runs that never matched would only lower it.
    """
    if not k or k <= SEISMO_BUDGET:
        return ""
    return round(llm_total / (k - SEISMO_BUDGET), 6)


def summarise(rows: list[dict], methods: list[str]) -> list[dict]:
    """One row per method, with both medians and the two break-even thresholds."""
    summary = []
    for method in methods:
        mine = [r for r in rows if r["method"] == method]
        if not mine:
            continue
        matched = [int(r["calls_to_match"]) for r in mine if r["matched"]]
        n_censored = len(mine) - len(matched)
        # Censored runs never matched, so they sit above every observed value.
        # Ranking them last keeps the median valid while fewer than half the
        # runs are censored, and sends it to infinity once more than half are.
        median = statistics.median(sorted(matched) + [float("inf")] * n_censored)
        # The matched-only median describes the runs that did match and is the
        # one an interquartile range belongs to.
        matched_median = statistics.median(matched) if matched else None
        quartiles = statistics.quantiles(matched, n=4) if len(matched) > 1 else []
        targets_never = sum(
            1
            for t in {r["target_index"] for r in mine}
            if not any(r["matched"] for r in mine if r["target_index"] == t)
        )
        horizons = sorted({r["budget"] for r in mine})
        seconds = [float(r["seconds"]) for r in mine if r["seconds"] != ""]
        machines = sorted({r["machine"] for r in mine if r["machine"]})
        summary.append(
            {
                "method": method,
                "horizon": horizons[0] if len(horizons) == 1 else ",".join(map(str, horizons)),
                "n_runs": len(mine),
                "n_matched": len(matched),
                "n_never_matched": n_censored,
                "n_targets_never_matched": targets_never,
                "median_calls_to_match": median,
                "matched_only_median": matched_median if matched else "",
                "matched_only_q1": quartiles[0] if quartiles else "",
                "matched_only_q3": quartiles[2] if quartiles else "",
                "ratio_to_seismo_50": (median / SEISMO_BUDGET) if median != float("inf") else "",
                "median_seconds_per_run": round(statistics.median(seconds), 1) if seconds else "",
                "machine": ",".join(machines),
                "breakeven_usd_per_oracle_call": breakeven(matched_median, SEISMO_LLM_COST_USD),
                "breakeven_seconds_per_oracle_call": breakeven(
                    matched_median, SEISMO_LLM_SECONDS_PER_ITER * SEISMO_BUDGET
                ),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_summary(summary: list[dict]) -> None:
    print(
        f"\n{'method':10s} {'runs':>5s} {'matched':>8s} {'median calls':>13s} "
        f"{'xSEISMO':>8s} {'USD/call':>12s} {'s/call':>10s}"
    )
    for row in summary:
        median = row["median_calls_to_match"]
        shown = "never" if median == float("inf") else f"{median:.0f}"
        ratio = row["ratio_to_seismo_50"]
        usd = row["breakeven_usd_per_oracle_call"]
        secs = row["breakeven_seconds_per_oracle_call"]
        print(
            f"{row['method']:10s} {row['n_runs']:5d} {row['n_matched']:8d} "
            f"{shown:>13s} {(f'{ratio:.0f}' if ratio != '' else '-'):>8s} "
            f"{(f'{usd:.4f}' if usd != '' else 'n/a'):>12s} "
            f"{(f'{secs:.3f}' if secs != '' else 'n/a'):>10s}"
        )


def analyse(args) -> None:
    task_dir = args.out / "sim_qed_pubchem"
    rows = per_run_rows(task_dir, args.methods, seismo_best_per_target(args.targets))
    if not rows:
        raise SystemExit(f"no baseline run files found under {task_dir}")
    summary = summarise(rows, args.methods)

    runs_csv = write_csv(args.out / "iso_quality_runs.csv", rows)
    summary_csv = write_csv(args.out / "iso_quality_summary.csv", summary)
    tex_path = args.out / "iso_quality_table.tex"
    tex_path.write_text(latex_table(summary))

    print_summary(summary)
    print(f"\nper-run detail: {runs_csv}")
    print(f"summary:        {summary_csv}")
    print(f"appendix table: {tex_path}")


def latex_table(summary: list[dict]) -> str:
    """The appendix table, generated from the run files so it cannot drift."""
    caption = (
        "\\caption{Oracle calls each baseline needs before its best-so-far molecule "
        f"matches the best molecule SEISMO found within {SEISMO_BUDGET} calls, on the "
        "Sim--QED task. Baselines run at their upstream defaults with three replicates "
        "on each of the 20 references. Horizons differ by method because cost per run "
        "grows sharply with the horizon. The median ranks runs that never match last, "
        "so where those are a large fraction it is a lower bound and is marked "
        "$\\geq$, and where they are the majority the median is not observed within "
        "the horizon and is marked $>$. The interquartile range is taken over the "
        "runs that did match. Break-even is the oracle cost above which SEISMO "
        "reaches the same "
        "molecule for less, against a measured "
        f"{SEISMO_LLM_COST_USD}~USD and "
        f"{SEISMO_LLM_SECONDS_PER_ITER * SEISMO_BUDGET:.0f}~s of LLM inference per "
        "SEISMO run.}"
    )
    header = (
        "Method & Horizon & Matched & Median calls & IQR of matched & Break-even & "
        "Break-even\\\\"
    )
    lines = [
        "% Generated by scripts/baselines/iso_quality.py, do not edit by hand.",
        "\\begin{table}[ht]",
        "\\centering",
        caption,
        "\\label{tab:appendix-iso-quality}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        header,
        " & (calls) & (of runs) & to match & runs & (USD/call) & (s/call)\\\\",
        "\\midrule",
    ]
    for row in summary:
        median = row["median_calls_to_match"]
        # round, not truncate, so the table agrees with the printed summary.
        # When over half the runs never match, the median is not observed at
        # all, so the only honest statement is that it exceeds the horizon.
        median_shown = (
            f"$>${row['horizon']:,}"
            if median == float("inf")
            else f"{round(median):,}"
        )
        iqr = (
            f"{round(row['matched_only_q1']):,}--{round(row['matched_only_q3']):,}"
            if row["matched_only_q1"] != ""
            else "--"
        )
        usd = row["breakeven_usd_per_oracle_call"]
        secs = row["breakeven_seconds_per_oracle_call"]
        # A median computed with many censored runs understates the truth, so
        # mark it as a bound rather than present it as an estimate.
        censored_fraction = row["n_never_matched"] / row["n_runs"] if row["n_runs"] else 0
        if median != float("inf") and censored_fraction >= 0.25:
            median_shown = f"$\\geq${median_shown}"
        lines.append(
            f"{PRETTY.get(row['method'], row['method'])} & "
            f"{row['horizon']:,} & "
            f"{row['n_matched']}/{row['n_runs']} & {median_shown} & {iqr} & "
            f"{(f'{usd:.4f}' if usd != '' else '--')} & "
            f"{(f'{secs:.3f}' if secs != '' else '--')}\\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def check_against_grid(args) -> None:
    """Replicate 1 shares its seed with the 50 call grid, so compare the two.

    Compares the exact sequence of the first 50 molecules, not the best score
    among them. Comparing only the best is too weak: two runs that sample
    different molecules can still share a top scorer and so appear to agree.
    """
    grid_dir = ROOT / "data/results/baseline_grid/sim_qed_pubchem"
    if not grid_dir.is_dir():
        print("\nno 50 call grid results to check against, skipping")
        return

    def prefix(trace: list[dict]) -> list[str]:
        ordered = sorted(trace, key=lambda e: int(e["iteration"]))
        return [e["smiles"] for e in ordered if int(e["iteration"]) <= SEISMO_BUDGET]

    for method in args.methods:
        checked = nested = 0
        overlaps = []
        for path in run_files(args.out / "sim_qed_pubchem" / method, "mol*_rep1"):
            run = load_run(path)
            grid_path = grid_dir / method / "default" / f"mol{run['target_index']}.json"
            if not grid_path.exists():
                continue
            mine = prefix(run["trace"])
            theirs = prefix(load_run(grid_path)["trace"])
            checked += 1
            if mine == theirs:
                nested += 1
            else:
                overlaps.append(len(set(mine) & set(theirs)))
        if not checked:
            continue
        if nested == checked:
            print(
                f"\n{method}: all {checked} replicate-1 runs reproduce the 50 call "
                f"grid exactly, so they are extensions of it"
            )
        else:
            mean_overlap = sum(overlaps) / len(overlaps)
            print(
                f"\n{method}: {nested}/{checked} replicate-1 runs reproduce the 50 "
                f"call grid. The rest share only {mean_overlap:.0f} of "
                f"{SEISMO_BUDGET} molecules on average, so for this method the "
                f"long-horizon run is a separate run, not an extension. For GP-BO "
                f"the cause is not the budget: its initial sample is seed-stable, "
                f"but mol-opt evaluates that sample in Python set iteration order, "
                f"so which molecules a short run scores depends on PYTHONHASHSEED. "
                f"Report it as such."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molopt-root", type=Path)
    parser.add_argument("--zinc-pool", type=Path)
    parser.add_argument("--budget", type=int, default=10000, help="long-horizon oracle budget")
    parser.add_argument("--reps", type=int, default=3, help="replicates per target")
    parser.add_argument("--targets", type=int, default=20)
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="comma-separated subset of " + ", ".join(METHODS),
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data/results/iso_quality")
    parser.add_argument(
        "--run-indices",
        default=None,
        help="comma-separated 1-based target indices, to shard across processes",
    )
    parser.add_argument(
        "--rep-indices",
        default=None,
        help="comma-separated 1-based replicate indices, to shard a target further",
    )
    parser.add_argument("--skip-analysis", action="store_true", help="run only, do not analyse")
    parser.add_argument("--analysis-only", action="store_true", help="analyse existing run files")
    args = parser.parse_args()

    args.methods = [m.strip() for m in args.methods.split(",")]
    unknown = [m for m in args.methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; choose from {METHODS}")

    if not args.analysis_only:
        if args.molopt_root is None or args.zinc_pool is None:
            raise SystemExit("--molopt-root and --zinc-pool are required unless --analysis-only")
        molopt_root = args.molopt_root.resolve()
        if not (molopt_root / "molopt").is_dir():
            raise SystemExit(f"no molopt package under {molopt_root}")
        sys.path.insert(0, str(molopt_root))
        install_tdc_stub(args.zinc_pool.resolve())
        pool = read_smiles_pool(args.zinc_pool.resolve())
        execute(args, pool)

    if not args.skip_analysis:
        analyse(args)
        check_against_grid(args)


if __name__ == "__main__":
    main()
