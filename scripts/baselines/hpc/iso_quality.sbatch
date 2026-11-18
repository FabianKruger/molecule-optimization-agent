#!/bin/bash
# Long-horizon iso-quality baseline runs, one array task per single run.
#
#     sbatch --time=0:30:00 --mem=4G scripts/baselines/hpc/iso_quality.sbatch graph_ga 10000
#     sbatch --time=2:00:00 --mem=4G scripts/baselines/hpc/iso_quality.sbatch reinvent 10000
#     sbatch                         scripts/baselines/hpc/iso_quality.sbatch gpbo     10000
#
# The method is the first argument and the oracle budget the second. All three
# can be in the queue at once, they write to the same directory but different
# files.
#
# The --time and --mem flags override the #SBATCH defaults below. They have to
# be given on the command line rather than worked out inside the script,
# because Slurm reads the directives before the script ever runs. The defaults
# are sized for GP BO, so it needs no overrides at all.
#
# A single run cannot be parallelised, because each oracle call depends on the
# result of the previous one. One (target, replicate) pair is therefore the
# smallest unit of work, and 20 targets by 3 replicates gives 60 independent
# tasks per method. Wall clock is then the cost of one run rather than three.
#
# If the cluster limits how many array tasks may run at once, or you want to be
# polite to other users, throttle with a percentage: --array=1-60%20 runs at
# most 20 at a time and finishes in the same total core-hours.
#
# Account and partition are site-specific, so they ship commented out. Fill in
# the two directives below and they apply to every submission, which is the
# least error-prone option if you always use the same cluster. Being a tracked
# file, your values will then show up as a local modification to it.
#
# The alternatives, if you would rather leave the file alone, are to export
#
#     export SBATCH_ACCOUNT=<your account>
#     export SBATCH_PARTITION=<your partition>
#
# or to pass -A and -p to sbatch. Precedence is command line, then environment,
# then the directives in this file, so an exported SBATCH_PARTITION silently
# beats a directive set here.

#SBATCH --job-name=iso_quality
##SBATCH --account=
##SBATCH --partition=
#SBATCH --array=1-60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=6:00:00
#SBATCH --output=logs/iso_quality_%x_%A_%a.out
#SBATCH --error=logs/iso_quality_%x_%A_%a.err

set -euo pipefail

# Arguments, not environment variables, so that they show up in squeue and do
# not depend on how the site configures --export.
METHOD="${1:-${METHOD:-}}"
BUDGET="${2:-${BUDGET:-10000}}"
REPS="${REPS:-3}"
TARGETS="${TARGETS:-20}"

case "$METHOD" in
    graph_ga|reinvent|gpbo) ;;
    "") echo "usage: sbatch $0 <graph_ga|reinvent|gpbo> [budget]" >&2; exit 1 ;;
    *)  echo "error: unknown method '$METHOD', expected graph_ga, reinvent or gpbo" >&2; exit 1 ;;
esac

TASK="${SLURM_ARRAY_TASK_ID:?this script expects to run as a Slurm array}"
if (( TASK < 1 || TASK > TARGETS * REPS )); then
    echo "error: array index $TASK is outside 1..$((TARGETS * REPS))" >&2
    exit 1
fi

# Task 1..60 maps onto (target 1..20, replicate 1..3).
TARGET=$(( (TASK - 1) / REPS + 1 ))
REP=$(( (TASK - 1) % REPS + 1 ))

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
PARENT="$(dirname "$REPO")"
cd "$REPO"

export PATH="$HOME/.pixi/bin:$PATH"

# One run is single-threaded by design, and one core is all the task is given.
# Letting the numerical libraries start their own thread pools would make tasks
# on the same node fight for cores and slow everything down.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# GP BO evaluates its initial sample in Python set iteration order, so without
# this the molecules a run scores depend on the interpreter's per-process string
# hash randomisation rather than on the run seed. Fixing it makes these runs
# exactly reproducible. The 4,000 call runs were produced without it.
export PYTHONHASHSEED=0

OUT="data/results/iso_quality_${BUDGET}"

echo "host        $(hostname)"
echo "method      $METHOD"
echo "budget      $BUDGET"
echo "task        $TASK of $((TARGETS * REPS))  ->  target $TARGET, replicate $REP"
echo "output      $OUT"
echo "started     $(date -Is)"

# An already-finished run is skipped, so resubmitting the same array after a
# timeout only redoes what is missing.
pixi run -e baselines python scripts/baselines/iso_quality.py \
    --molopt-root "$PARENT/mol-opt" \
    --zinc-pool "$PARENT/zinc_250k.tab" \
    --budget "$BUDGET" \
    --reps "$REPS" \
    --targets "$TARGETS" \
    --methods "$METHOD" \
    --out "$OUT" \
    --run-indices "$TARGET" \
    --rep-indices "$REP" \
    --skip-analysis

echo "finished    $(date -Is)"
