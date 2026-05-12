#!/bin/zsh
#
# TDC Tasks Experiments Runner for Zero-Shot Generation Baseline
# This script runs molecule optimization experiments for multiple TDC oracles.
# Each oracle gets its own results folder with JSON files from repetitions.
#

set -e  # Exit on error

# List of TDC oracle names to run experiments for
ORACLE_NAMES=(
    # Basic oracles
    "qed"
    "drd2"
    "gsk3b"
    "jnk3"
    # Rediscovery tasks
    "celecoxib_rediscovery"
    "troglitazone_rediscovery"
    "thiothixene_rediscovery"
    # Similarity tasks
    "albuterol_similarity"
    "mestranol_similarity"
    # Isomer tasks
    "isomers_c7h8n2o2"
    "isomers_c9h10n2o2pf2cl"
    # Median molecules
    "median1"
    "median2"
    # MPO tasks
    "osimertinib_mpo"
    "fexofenadine_mpo"
    "ranolazine_mpo"
    "perindopril_mpo"
    "amlodipine_mpo"
    "sitagliptin_mpo"
    "zaleplon_mpo"
    # SMARTS-constrained
    "valsartan_smarts"
    # Hop tasks
    "deco_hop"
    "scaffold_hop"
)

# Number of molecules to generate per task
NUMBER_OF_MOLECULES=50

# LLM model name
LLM_MODEL="claude-opus-4.5"

# LLM temperature
TEMPERATURE=1

# Number of repetitions per oracle
NUM_REPETITIONS=3

# Zero-shot generation mode (batch or independent)
MODE="batch"

'''
for run_idx in $(seq 1 "$NUM_REPETITIONS"); do
    pixi run python scripts/baselines/zero-shot-generation.py \
        --n-molecules $NUMBER_OF_MOLECULES \
        --model $LLM_MODEL \
        --temperature $TEMPERATURE \
        --tasks "${ORACLE_NAMES[@]}" \
        --mode $MODE
done
'''

PROMPT_STYLE=(
    "task"
    "generic"
)
    
for prompt_style in "${PROMPT_STYLE[@]}"; do
    for run_idx in $(seq 1 "$NUM_REPETITIONS"); do
        pixi run python scripts/baselines/zero-shot-generation.py \
            --n-molecules $NUMBER_OF_MOLECULES \
            --model $LLM_MODEL \
            --temperature $TEMPERATURE \
            --tasks "ic50mpro_qed_novel" \
            --mode $MODE \
            --prompt-style $prompt_style
    done

    pixi run python scripts/baselines/zero-shot-generation.py \
        --n-molecules $NUMBER_OF_MOLECULES \
        --model $LLM_MODEL \
        --temperature $TEMPERATURE \
        --tasks "similarity_qed_pubchem" \
        --mode $MODE \
        --prompt-style $prompt_style
done

