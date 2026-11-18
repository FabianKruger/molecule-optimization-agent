# Long-horizon baseline runs on a Slurm cluster

Runs the iso-quality baselines past the horizons reached locally, so that every
method can be reported at the same 10,000 oracle call horizon. Graph-GA is
already there. REINVENT and GP BO are not, and GP BO is the reason this needs a
cluster: roughly 100 to 140 core-hours for its 60 runs.

## How the work splits

A single run cannot be parallelised: each oracle call picks its next molecule
from the results of the previous ones. So the unit of work is one
(target, replicate) pair, and 20 targets by 3 replicates gives **60 independent
single-core tasks per method**. Both methods can sit in the queue at once, so
the experiment saturates 120 cores and the wall clock is the cost of one run,
not of a queue.

Estimated cost per run at 10,000 calls, extrapolated from probes at 2,000 and
4,000 calls. REINVENT scales as about $n^{1.6}$ and GP BO as about $n^{2.6}$,
the latter because it refits a Gaussian process whose cost is cubic in the
number of observations.

| Method | Per run | 60 runs | Wall clock, 60 tasks at once |
|---|---|---|---|
| Graph-GA | about 1 min | about 1 core-hour | about 1 min |
| REINVENT | 16 to 24 min | 16 to 24 core-hours | 16 to 24 min |
| GP BO | 1.6 to 2.3 h | 99 to 140 core-hours | 1.6 to 2.3 h |

Submit all three and the whole thing is done in a couple of hours on 180 cores.

Graph-GA has already been run at 10,000 calls elsewhere, and it is included here
anyway because it costs about one core-hour and it makes every reported timing
comparable. Mixing hardware between methods would otherwise undercut the one
claim in this analysis that is about time rather than oracle calls.

Treat the GP BO figure as indicative. It is a two-point extrapolation, and a
previous extrapolation of GP BO's cost was wrong by a factor of five on the
optimistic side. The array structure makes that cheap to absorb: one task is one
run, so a timeout loses one run and resubmitting the array redoes only what is
missing.

If the cluster caps concurrent array tasks, or you would rather not take 120
cores at once, throttle with `--array=1-60%20`. That runs at most 20 at a time
for the same total core-hours.

Memory is per task, not per node: GP BO peaked at 2.2 GB resident at this
horizon, so 6 GB per task is comfortable. Nothing here needs a GPU.

## What you need on the cluster

Two repositories side by side, plus the starting pool, all fetched for you:

```
<somewhere>/
  molecule-optimization-agent/     this repository, branch paper-revision
  mol-opt/                         cloned by setup.sh, pinned commit
  zinc_250k.tab                    downloaded by setup.sh, md5 checked
```

Nothing has to be copied by hand. `setup.sh` downloads the 249,455 molecule ZINC
250k pool straight from the Harvard Dataverse file TDC points at for
`MolGen(name="ZINC")`, so pytdc is not needed, and verifies its md5 is
`9e4754d72db297d496def3498a926979` before going on. A wrong checksum stops the
setup, because a different starting pool would silently invalidate the
comparison against the runs already reported.

The pool is deliberately not in git. ZINC is free to use, but redistributing a
significant subset of it needs written permission from its authors.

`setup.sh` needs outbound network, so run it on a login node. If your compute
cluster has no outbound network at all, it says so and tells you where to put
the file.

## Setup, once

```bash
git clone -b paper-revision https://github.com/FabianKruger/molecule-optimization-agent.git
cd molecule-optimization-agent
bash scripts/baselines/hpc/setup.sh
```

This installs pixi into `~/.pixi` if it is not already available, clones mol-opt
at the pinned commit beside this repository, builds the `baselines` environment
from `pixi.lock`, and verifies the pool. Run it on a login node, since it needs
network access.

mol-opt is used unpatched. The compatibility shims it needs on a current
scientific stack are applied at import time by `hyperparameter_grid.py`, so
nothing in that clone should be edited.

## Submit

Account and partition are site-specific and are not in the batch file. Export
them once and Slurm picks them up, or pass `-A` and `-p` to `sbatch`:

```bash
export SBATCH_ACCOUNT=<your account>
export SBATCH_PARTITION=<your partition>
```

Then, from the repository root:

```bash
METHOD=graph_ga BUDGET=10000 sbatch --time=0:30:00 --mem=4G \
    scripts/baselines/hpc/iso_quality.sbatch

METHOD=reinvent BUDGET=10000 sbatch --time=2:00:00 --mem=4G \
    scripts/baselines/hpc/iso_quality.sbatch

METHOD=gpbo BUDGET=10000 sbatch \
    scripts/baselines/hpc/iso_quality.sbatch
```

Sixty array tasks per method, one run each, one core each. Array task $i$ maps
onto target $\lfloor (i-1)/3 \rfloor + 1$ and replicate $((i-1) \bmod 3) + 1$,
so tasks 1 to 3 are the three replicates of target 1. The defaults in the batch
file (6 h, 6 GB) are sized for GP BO. REINVENT needs much less, hence the
overrides above.

Results go to `data/results/iso_quality_10000/`, which does not touch the
existing 4,000 call results in `data/results/iso_quality/`.

Finished runs are skipped on resubmission, so if a task hits its time limit,
raise `--time` and submit the same command again.

One deliberate difference from the 4,000 call runs: the batch file sets
`PYTHONHASHSEED=0`. GP BO evaluates its initial sample in Python set iteration
order, so without this the molecules a run scores depend on per-process string
hash randomisation rather than on the run seed. These runs are therefore exactly
reproducible where the earlier ones are not.

## Collect and push

```bash
bash scripts/baselines/hpc/collect.sh 10000
```

This reports how many of the 60 runs each method produced, validates every run
file, writes the per-run CSV, the summary CSV and the appendix table, gzips the
traces, commits to a branch `iso-quality-10000` and pushes it.

Each run records its own wall-clock time and `platform.machine()`, so the
per-run CSV carries `seconds` and `machine` and the summary carries
`median_seconds_per_run`. Runs are single-threaded, so that time is also their
core time. Earlier runs have these columns empty, since they predate the
recording.

Compression matters here: a 10,000 call trace is 1.3 MB plain and 0.22 MB
gzipped, so 120 runs travel as about 27 MB instead of 155 MB. `iso_quality.py`
and `validate_runs.py` both read either form.

## Pull the results back

```bash
git fetch origin
git checkout iso-quality-10000
pixi run -e baselines python scripts/baselines/iso_quality.py \
    --analysis-only --out data/results/iso_quality_10000
```

Run the analysis again here rather than trusting the cluster's copy.
`check_against_grid` compares each replicate 1 run against the 50 call
hyperparameter grid. If that grid is committed, the check runs on the cluster
too and doubles as a cross-platform test: a replicate 1 run that reproduced the
grid locally but not on the cluster tells you the processor changed the
trajectory. If the grid is not committed, the check reports itself skipped.

## Why rerun Graph-GA

Two things come out of these runs, and they behave differently.

The oracle-call counts are fixed by the run seed, so for Graph-GA and REINVENT
the cluster should reproduce what was measured elsewhere. It may not do so
exactly: the earlier runs were on arm64 and the cluster is presumably x86_64, and
the oracles are floating-point, so a difference in the last bits can change which
candidate a generation selects and send the trajectories apart. Nothing about the
method changes, but an individual number might.

The timings are hardware. The analysis makes one claim about time, that the most
sample-efficient method per oracle call is also the most expensive per call in
computation, and that claim is only sound if all three methods are timed on the
same processor. Graph-GA costs about one core-hour to redo, so there is no reason
to leave it as the odd one out.

## Scope

Sim--QED only, as with the 4,000 call runs. The IC50--QED--Novel task issues a
live PubChem query per oracle call, so a 10,000 call horizon would mean hundreds
of thousands of queries and is out of the question.
