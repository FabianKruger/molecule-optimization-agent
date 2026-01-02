#!/bin/bash
#
# XAI Experiments Runner
# This script runs molecule optimization experiments with different XAI settings.
#

set -e  # Exit on error

# ==============================================================================
# HYPERPARAMETERS - Modify these values as needed
# ==============================================================================

# Molecule SMILES for similarity experiments
relevant_molecule="O=C1c3c(O/C(=C1/O)c2ccc(O)c(O)c2)cc(O)cc3O"  # Quercetin

# Similarity + QED objective thresholds
sim_threshold=0.8          # min_similarity
qed_threshold_sim=0.8      # min_qed for similarity_qed
target_score=0.8          # target combined score

# General experiment settings
max_iterations=51
llm_model="claude-opus-4.5"
temperature=0.1
recursion_limit=300

# Number of repetitions per experiment
num_repetitions=5

# ==============================================================================
# PATHS - Relative to molecule-optimization-agent directory
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_BASE_DIR="$REPO_DIR/data/results"
TEMP_CONFIG_DIR="$REPO_DIR/.temp_configs_quercetin"

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
recursion_limit: ${recursion_limit}

llm:
  model: ${llm_model}
  temperature: ${temperature}

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
    target_score: ${target_score}
    min_similarity: ${sim_threshold}
    min_qed: ${qed_threshold_sim}
    max_iterations: ${max_iterations}
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

# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

echo "=============================================="
echo "XAI Experiments Runner"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Relevant molecule: $relevant_molecule"
echo "  Similarity threshold: $sim_threshold"
echo "  QED threshold: $qed_threshold_sim"
echo "  Target score: $target_score"
echo "  Max iterations: $max_iterations"
echo "  LLM model: $llm_model"
echo "  Temperature: $temperature"
echo "  Repetitions: $num_repetitions"
echo ""

# Create temporary config directory
create_temp_config_dir

# Trap to cleanup on exit
trap cleanup_temp_configs EXIT


# Get XAI directory name
# Args: $1=xai_mode
get_xai_dir_name() {
    local xai_mode="$1"
    if [ "$xai_mode" == "none" ]; then
        echo "no_xai"
    elif [ "$xai_mode" == "partial" ]; then
        echo "partial_xai"
    elif [ "$xai_mode" == "no_description" ]; then
        echo "no_description"
    else
        echo "full_xai"
    fi
}

echo "=============================================="
echo "EXPERIMENT: similarity_qed - relevant_molecule"
echo "=============================================="

for xai_mode in "partial"; do # "full" "none" "partial" "no_description"
    XAI_DIR_NAME=$(get_xai_dir_name "$xai_mode")
    xai_dir_name="${llm_model}_${XAI_DIR_NAME}_${target_score}"
    log_dir="$RESULTS_BASE_DIR/similarity_qed_quercetin/$xai_dir_name"
    mkdir -p "$log_dir"
    
    echo ""
    echo "XAI mode: $xai_mode ($xai_dir_name)"
    
    for rep in $(seq 1 $num_repetitions); do
        config_path="$TEMP_CONFIG_DIR/sim_qed_relevant_${xai_mode}_${rep}.yaml"
        experiment_name="sim_qed_relevant_${xai_mode}_rep${rep}"
        
        create_similarity_qed_config "$config_path" "$experiment_name" "$log_dir" "$relevant_molecule" "$xai_mode"
        run_experiment "$config_path" "Repetition $rep / $num_repetitions"
    done
done

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "=============================================="
echo ""
echo "Results saved to: $log_dir"
