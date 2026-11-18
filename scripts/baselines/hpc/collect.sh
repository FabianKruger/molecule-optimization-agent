#!/bin/bash
# Gzip the finished runs, analyse them, and push everything to a results branch.
# Run from the repository root once the array jobs are done:
#
#     bash scripts/baselines/hpc/collect.sh 10000
#
# A 10,000 call trace is 1.3 MB plain and 0.22 MB gzipped, so 120 runs travel
# as roughly 27 MB rather than 155 MB. iso_quality.py reads either form.
set -euo pipefail

BUDGET="${1:-10000}"
BRANCH="${BRANCH:-iso-quality-${BUDGET}}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
export PATH="$HOME/.pixi/bin:$PATH"

OUT="data/results/iso_quality_${BUDGET}"
if [[ ! -d "$OUT" ]]; then
    echo "error: no results under $OUT" >&2
    exit 1
fi

# --- how complete is it? ----------------------------------------------------
echo "runs found:"
for d in "$OUT"/sim_qed_pubchem/*/; do
    [[ -d "$d" ]] || continue
    n=$(find "$d" -name 'mol*_rep*.json' -o -name 'mol*_rep*.json.gz' | wc -l)
    printf "  %-10s %s of 60\n" "$(basename "$d")" "$n"
done

# --- check every run before trusting it -------------------------------------
echo
echo "validating"
pixi run -e baselines python scripts/baselines/validate_runs.py --out "$OUT"

# --- analyse ----------------------------------------------------------------
# If the 50 call grid is not in the clone, the nesting check against it reports
# itself skipped and has to be run after pulling. Everything else is computed.
echo
echo "analysing"
pixi run -e baselines python scripts/baselines/iso_quality.py \
    --analysis-only --out "$OUT"

# --- compress ---------------------------------------------------------------
echo
echo "gzipping traces"
find "$OUT" -name 'mol*_rep*.json' -print0 | xargs -0 -r gzip -f
du -sh "$OUT"

# --- push -------------------------------------------------------------------
echo
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
git add "$OUT"
if git diff --cached --quiet; then
    echo "nothing new to commit"
else
    git commit -m "Add iso-quality baseline runs at a ${BUDGET} call horizon

Produced on the cluster with scripts/baselines/hpc/iso_quality.sbatch, one
array task per target, three replicates each, single-threaded with
PYTHONHASHSEED=0 so GP BO's initial evaluation order is reproducible.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
fi
git push -u origin "$BRANCH"

echo
echo "pushed to $BRANCH. Pull it on the other machine with:"
echo "  git fetch origin && git checkout $BRANCH"
echo "  pixi run -e baselines python scripts/baselines/iso_quality.py --analysis-only --out $OUT"
