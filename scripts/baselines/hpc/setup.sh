#!/bin/bash
# One-time setup on the cluster login node. Run from the repository root:
#
#     bash scripts/baselines/hpc/setup.sh
#
# Installs pixi into ~/.pixi if it is not already there, clones mol-opt beside
# this repository at the pinned commit, builds the baselines environment, and
# downloads the ZINC starting pool. Needs outbound network, so run it on a
# login node. Everything it fetches is pinned or checksummed, so rerunning it
# is safe and it skips whatever is already in place.
set -euo pipefail

MOLOPT_COMMIT=63382d78890e910080ef9a9b3b6d04a4552aff85
# TDC's own Dataverse file for MolGen(name="ZINC"), file id from tdc/metadata.py.
ZINC_URL=https://dataverse.harvard.edu/api/access/datafile/4170963
ZINC_MD5=9e4754d72db297d496def3498a926979

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PARENT="$(dirname "$REPO")"
cd "$REPO"

if [[ ! -f pyproject.toml ]]; then
    echo "error: $REPO does not look like the repository root" >&2
    exit 1
fi

# --- pixi -------------------------------------------------------------------
if ! command -v pixi >/dev/null 2>&1; then
    if [[ -x "$HOME/.pixi/bin/pixi" ]]; then
        export PATH="$HOME/.pixi/bin:$PATH"
    else
        echo "installing pixi into ~/.pixi"
        curl -fsSL https://pixi.sh/install.sh | bash
        export PATH="$HOME/.pixi/bin:$PATH"
    fi
fi
echo "pixi: $(command -v pixi) $(pixi --version)"

# --- mol-opt ----------------------------------------------------------------
if [[ ! -d "$PARENT/mol-opt/.git" ]]; then
    echo "cloning mol-opt into $PARENT/mol-opt"
    git clone https://github.com/wenhao-gao/mol-opt.git "$PARENT/mol-opt"
fi
git -C "$PARENT/mol-opt" fetch --quiet origin
git -C "$PARENT/mol-opt" checkout --quiet "$MOLOPT_COMMIT"
echo "mol-opt: pinned at $MOLOPT_COMMIT"

# mol-opt is used unpatched. The compatibility shims it needs on a current
# scientific stack are applied at import time by hyperparameter_grid.py, so
# nothing in that clone should ever be edited.
if [[ -n "$(git -C "$PARENT/mol-opt" status --porcelain)" ]]; then
    echo "warning: the mol-opt clone has local modifications, which it should not" >&2
fi

# --- environment ------------------------------------------------------------
echo "building the baselines environment, which takes a few minutes"
pixi install -e baselines

# --- ZINC pool --------------------------------------------------------------
# The 249,455 molecule ZINC 250k starting pool, which is what TDC serves as
# MolGen(name="ZINC"). Downloaded straight from the Harvard Dataverse file TDC
# points at, so pytdc itself is not needed. It is deliberately not in git: ZINC
# is free to use, but redistributing a significant subset of it needs written
# permission from its authors.
POOL="$PARENT/zinc_250k.tab"
if [[ ! -f "$POOL" ]]; then
    echo "downloading the ZINC pool, 11.8 MB"
    if ! curl -fsSL "$ZINC_URL" -o "$POOL.partial"; then
        rm -f "$POOL.partial"
        cat >&2 <<EOF

error: could not download the ZINC pool from

    $ZINC_URL

If the compute cluster has no outbound network even on the login node, fetch it
somewhere that does and copy it to

    $POOL

EOF
        exit 1
    fi
    mv "$POOL.partial" "$POOL"
fi
GOT=$(md5sum "$POOL" | cut -d' ' -f1)
if [[ "$GOT" != "$ZINC_MD5" ]]; then
    echo "error: ZINC pool checksum is $GOT, expected $ZINC_MD5" >&2
    echo "delete $POOL and rerun to download it again" >&2
    exit 1
fi
echo "ZINC pool: $POOL, $(wc -l < "$POOL") lines, checksum matches"

mkdir -p "$REPO/logs"
echo
echo "setup complete. Submit with:"
echo "  METHOD=reinvent BUDGET=10000 sbatch --time=2:00:00 --mem=4G scripts/baselines/hpc/iso_quality.sbatch"
echo "  METHOD=graph_ga BUDGET=10000 sbatch --time=0:30:00 --mem=4G scripts/baselines/hpc/iso_quality.sbatch"
echo "  METHOD=gpbo     BUDGET=10000 sbatch                        scripts/baselines/hpc/iso_quality.sbatch"
