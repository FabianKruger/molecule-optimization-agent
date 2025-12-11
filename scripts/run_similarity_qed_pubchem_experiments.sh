#!/bin/bash
#
# Similarity + QED Experiments Runner for PubChem Molecules
# This script runs molecule optimization experiments with Similarity + QED objective
# for each of the 20 random SMILES from PubChem.
#

set -e  # Exit on error

# ==============================================================================
# HYPERPARAMETERS - Modify these values as needed
# ==============================================================================

# XAI setting: "full", "partial", or "none"
XAI_MODE="full"

# LLM model name
LLM_MODEL="gpt-5.1"

# Results base directory (will be created if it doesn't exist)
RESULTS_BASE_DIR="data/results"

# Similarity + QED objective thresholds
MIN_SIMILARITY=0.7
MIN_QED=0.7
TARGET_SCORE=0.9
MAX_ITERATIONS=1

# General experiment settings
TEMPERATURE=0.0
RECURSION_LIMIT=300

# Number of repetitions per molecule
NUM_REPETITIONS=1

# ==============================================================================
# PATHS - Relative to molecule-optimization-agent directory
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SAMPLED_MOLECULES_FILE="$REPO_DIR/src/sampling_molecule_targets/sampled_molecules.txt"
TEMP_CONFIG_DIR="$REPO_DIR/.temp_configs"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

create_temp_config_dir() {
    mkdir -p "$TEMP_CONFIG_DIR"
}

cleanup_temp_configs() {
    rm -rf "$TEMP_CONFIG_DIR"
}

# Create a similarity_qed config file
# Args: $1=output_path, $2=experiment_name, $3=log_dir, $4=target_smiles, $5=xai_mode
create_similarity_qed_config() {
    local output_path="$1"
    local experiment_name="$2"
    local log_dir="$3"
    local target_smiles="$4"
    local xai_mode="$5"

    cat > "$output_path" << EOF
experiment_name: ${experiment_name}

log_dir: ${log_dir}
recursion_limit: ${RECURSION_LIMIT}

llm:
  model: ${LLM_MODEL}
  temperature: ${TEMPERATURE}

oracle:
  name: composite
  params:
    oracles:
      - name: similarity
        params:
          target_smiles: "${target_smiles}"
      - name: qed
        params: {}
    weights: [0.5, 0.5]
    names: ["Similarity", "QED"]

objective:
  name: similarity_qed
  params:
    target_score: ${TARGET_SCORE}
    min_similarity: ${MIN_SIMILARITY}
    min_qed: ${MIN_QED}
    max_iterations: ${MAX_ITERATIONS}
    xai: ${xai_mode}
EOF
}

# Run experiment
# Args: $1=config_path, $2=description
run_experiment() {
    local config_path="$1"
    local description="$2"
    
    echo "  Running: $description"
    cd "$REPO_DIR"
    pixi run molopt --config "$config_path"
}

# Get XAI directory name
# Args: $1=xai_mode
get_xai_dir_name() {
    local xai_mode="$1"
    if [ "$xai_mode" == "none" ]; then
        echo "no_xai"
    elif [ "$xai_mode" == "partial" ]; then
        echo "partial_xai"
    else
        echo "full_xai"
    fi
}

# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

echo "=============================================="
echo "Similarity + QED PubChem Experiments Runner"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  XAI mode: $XAI_MODE"
echo "  LLM model: $LLM_MODEL"
echo "  Min similarity: $MIN_SIMILARITY"
echo "  Min QED: $MIN_QED"
echo "  Target score: $TARGET_SCORE"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Temperature: $TEMPERATURE"
echo "  Repetitions per molecule: $NUM_REPETITIONS"
echo "  Sampled molecules file: $SAMPLED_MOLECULES_FILE"
echo ""

# Validate XAI mode
if [[ ! "$XAI_MODE" =~ ^(full|partial|none)$ ]]; then
    echo "ERROR: XAI_MODE must be 'full', 'partial', or 'none'"
    exit 1
fi

# Check if sampled molecules file exists
if [ ! -f "$SAMPLED_MOLECULES_FILE" ]; then
    echo "ERROR: Sampled molecules file not found: $SAMPLED_MOLECULES_FILE"
    exit 1
fi

# Create results directory with LLM name and XAI setting
XAI_DIR_NAME=$(get_xai_dir_name "$XAI_MODE")
RESULTS_DIR="$REPO_DIR/$RESULTS_BASE_DIR/similarity_qed_pubchem/${LLM_MODEL}_${XAI_DIR_NAME}"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Create temporary config directory
create_temp_config_dir

# Trap to cleanup on exit
trap cleanup_temp_configs EXIT

# Read SMILES from file and run experiments
MOLECULE_INDEX=0
TOTAL_EXPERIMENTS=0

echo "=============================================="
echo "Running experiments for each PubChem molecule"
echo "=============================================="
echo ""

while IFS= read -r target_smiles || [ -n "$target_smiles" ]; do
    # Skip empty lines
    if [ -z "$target_smiles" ]; then
        continue
    fi
    
    # Trim whitespace
    target_smiles=$(echo "$target_smiles" | xargs)
    
    MOLECULE_INDEX=$((MOLECULE_INDEX + 1))
    
    echo "----------------------------------------------"
    echo "Molecule $MOLECULE_INDEX / 20"
    echo "Target SMILES: $target_smiles"
    echo "----------------------------------------------"
    
    for rep in $(seq 1 $NUM_REPETITIONS); do
        TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
        
        # Create experiment name
        experiment_name="sim_qed_pubchem_mol${MOLECULE_INDEX}_${XAI_MODE}_rep${rep}"
        
        # Create config file
        config_path="$TEMP_CONFIG_DIR/${experiment_name}.yaml"
        create_similarity_qed_config "$config_path" "$experiment_name" "$RESULTS_DIR" "$target_smiles" "$XAI_MODE"
        
        # Run experiment
        run_experiment "$config_path" "Molecule $MOLECULE_INDEX, Repetition $rep / $NUM_REPETITIONS"
    done
    
    echo ""
done < "$SAMPLED_MOLECULES_FILE"

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "=============================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Total experiments run: $TOTAL_EXPERIMENTS"
echo ""
