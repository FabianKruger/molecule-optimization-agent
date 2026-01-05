#!/bin/bash
#
# XAI Experiments Runner
# This script runs molecule optimization experiments using XGBoost IC50 predictor
#

set -e  # Exit on error

# ==============================================================================
# HYPERPARAMETERS - Modify these values as needed
# ==============================================================================

# IC50 objectives thresholds
IC50_THRESHOLD=10.0        # target_ic50_nM
QED_THRESHOLD=0.6          # min_qed for ic50_qed_novel

# General experiment settings
MAX_ITERATIONS=51
LLM_MODEL="claude-opus-4.5"
TEMPERATURE=0.1
RECURSION_LIMIT=300

# Number of repetitions per experiment
NUM_REPETITIONS=3

# ==============================================================================
# PATHS - Relative to molecule-optimization-agent directory
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_BASE_DIR="$REPO_DIR/data/results"
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


# Create an ic50mpro_qed_novel config file
# Args: $1=output_path, $2=experiment_name, $3=log_dir, $4=xai_mode
create_ic50mpro_qed_novel_config() {
    local output_path="$1"
    local experiment_name="$2"
    local log_dir="$3"
    local xai_mode="$4"

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
      - name: ic50mpro
        params:
          model_name: "src/xgboost_training/xgb_maccs_best"
      - name: qed
        params: {}
      - name: novel
        params: {}
    weights: [1.0, 0.0, 0.0]
    names: ["IC50", "QED", "Novelty"]

objective:
  name: ic50mpro_qed_novel
  params:
    target_ic50_nM: ${IC50_THRESHOLD}
    min_qed: ${QED_THRESHOLD}
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

# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

echo "=============================================="
echo "XAI Experiments Runner"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  IC50 threshold: $IC50_THRESHOLD"
echo "  QED threshold: $QED_THRESHOLD"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  LLM model: $LLM_MODEL"
echo "  Temperature: $TEMPERATURE"
echo "  Repetitions: $NUM_REPETITIONS"
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
    elif [ "$xai_mode" == "no_description" ]; then
        echo "no_description"
    else
        echo "${xai_mode}_xai"
    fi
}

echo ""
echo "=============================================="
echo "EXPERIMENT: ic50mpro_qed_novel"
echo "=============================================="

for xai_mode in "no_description"; do #"full" "partial" "none"
    XAI_DIR_NAME=$(get_xai_dir_name "$xai_mode")
    xai_dir_name="${LLM_MODEL}_${XAI_DIR_NAME}"
    
    log_dir="$RESULTS_BASE_DIR/ic50mpro_qed/$xai_dir_name"
    mkdir -p "$log_dir"
    
    echo ""
    echo "XAI mode: $xai_mode ($xai_dir_name)"
    
    for rep in $(seq 1 $NUM_REPETITIONS); do
        config_path="$TEMP_CONFIG_DIR/ic50mpro_qed_novel_${xai_mode}_${rep}.yaml"
        experiment_name="ic50mpro_qed_novel_${xai_mode}_rep${rep}"
        
        create_ic50mpro_qed_novel_config "$config_path" "$experiment_name" "$log_dir" "$xai_mode"
        run_experiment "$config_path" "Repetition $rep / $NUM_REPETITIONS"
    done
done

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "=============================================="
echo ""
echo "Results saved to: $RESULTS_BASE_DIR"