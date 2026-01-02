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
ic50_threshold=20.0        # target_ic50_nM
qed_threshold_ic50=0.5     # min_qed for ic50_qed_novel

# General experiment settings
max_iterations=3
llm_model="claude-opus-4.5"
temperature=0.1
recursion_limit=300

# Number of repetitions per experiment
num_repetitions=1

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
recursion_limit: ${recursion_limit}

llm:
  model: ${llm_model}
  temperature: ${temperature}

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
    weights: [0.5, 0.5, 0.0]
    names: ["IC50", "QED", "Novelty"]

objective:
  name: ic50mpro_qed_novel
  params:
    target_ic50_nM: ${ic50_threshold}
    min_qed: ${qed_threshold_ic50}
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
echo "  IC50 threshold: $ic50_threshold"
echo "  QED threshold: $qed_threshold_ic50"
echo "  Max iterations: $max_iterations"
echo "  LLM model: $llm_model"
echo "  Temperature: $temperature"
echo "  Repetitions: $num_repetitions"
echo ""

# Create temporary config directory
create_temp_config_dir

# Trap to cleanup on exit
trap cleanup_temp_configs EXIT

echo ""
echo "=============================================="
echo "EXPERIMENT: ic50mpro_qed_novel"
echo "=============================================="

for xai_mode in "full" "partial" "none"; do
    xai_dir_name="${xai_mode}_xai"
    if [ "$xai_mode" == "none" ]; then
        xai_dir_name="no_xai"
    fi
    
    log_dir="$RESULTS_BASE_DIR/ic50mpro_qed/$xai_dir_name"
    mkdir -p "$log_dir"
    
    echo ""
    echo "XAI mode: $xai_mode ($xai_dir_name)"
    
    for rep in $(seq 1 $num_repetitions); do
        config_path="$TEMP_CONFIG_DIR/ic50mpro_qed_novel_${xai_mode}_${rep}.yaml"
        experiment_name="ic50mpro_qed_novel_${xai_mode}_rep${rep}"
        
        create_ic50mpro_qed_novel_config "$config_path" "$experiment_name" "$log_dir" "$xai_mode"
        run_experiment "$config_path" "Repetition $rep / $num_repetitions"
    done
done

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "=============================================="
echo ""
echo "Results saved to: $RESULTS_BASE_DIR"