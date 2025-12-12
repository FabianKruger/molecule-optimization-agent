#!/bin/bash
#
# TDC Tasks Experiments Runner
# This script runs molecule optimization experiments for multiple TDC oracles.
# Each oracle gets its own results folder with JSON files from repetitions.
#

set -e  # Exit on error

# ==============================================================================
# HYPERPARAMETERS - Modify these values as needed
# ==============================================================================

# List of TDC oracle names to run experiments for
# Examples: drd2, gsk3b, jnk3, qed, sa, logp, etc.
ORACLE_NAMES=(
    "drd2"
    "gsk3b"
    "jnk3"
)

# Per-oracle configuration: direction
# Format: ORACLE_DIRECTION["oracle_name"]="maximize" or "minimize"
declare -A ORACLE_DIRECTION

# Configure direction for each oracle
ORACLE_DIRECTION["drd2"]="maximize"
ORACLE_DIRECTION["gsk3b"]="maximize"
ORACLE_DIRECTION["jnk3"]="maximize"

# Maximum number of iterations per experiment
MAX_ITERATIONS=20

# Number of repetitions per oracle
NUM_REPETITIONS=10

# LLM model name
LLM_MODEL="gpt-5.1"

# LLM temperature
TEMPERATURE=0.3

# Recursion limit for LangGraph
RECURSION_LIMIT=100

# Results base directory (will be created if it doesn't exist)
RESULTS_BASE_DIR="results/tdc_tasks"

# ==============================================================================
# PATHS - Relative to molecule-optimization-agent directory
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
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

# Create a TDC tasks config file
# Args: $1=output_path, $2=experiment_name, $3=log_dir, $4=oracle_name, $5=direction, $6=max_iterations
create_tdc_config() {
    local output_path="$1"
    local experiment_name="$2"
    local log_dir="$3"
    local oracle_name="$4"
    local direction="$5"
    local max_iterations="$6"

    cat > "$output_path" << EOF
experiment_name: ${experiment_name}

log_dir: ${log_dir}
recursion_limit: ${RECURSION_LIMIT}

llm:
  model: ${LLM_MODEL}
  temperature: ${TEMPERATURE}

oracle:
  name: tdc_tasks
  params:
    oracle_name: ${oracle_name}

objective:
  name: tdc_tasks
  params:
    direction: ${direction}
    max_iterations: ${max_iterations}
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
echo "TDC Tasks Experiments Runner"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Oracle names: ${ORACLE_NAMES[*]}"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Repetitions per oracle: $NUM_REPETITIONS"
echo "  LLM model: $LLM_MODEL"
echo "  Temperature: $TEMPERATURE"
echo "  Results base directory: $RESULTS_BASE_DIR"
echo ""
echo "Per-oracle settings:"
for oracle_name in "${ORACLE_NAMES[@]}"; do
    direction="${ORACLE_DIRECTION[$oracle_name]}"
    echo "  $oracle_name: direction=$direction"
done
echo ""

# Validate oracle configurations
for oracle_name in "${ORACLE_NAMES[@]}"; do
    direction="${ORACLE_DIRECTION[$oracle_name]}"
    
    if [ -z "$direction" ]; then
        echo "ERROR: Direction not set for oracle '$oracle_name'"
        exit 1
    fi
    
    if [[ ! "$direction" =~ ^(maximize|minimize)$ ]]; then
        echo "ERROR: Invalid direction '$direction' for oracle '$oracle_name'. Must be 'maximize' or 'minimize'"
        exit 1
    fi
done

# Validate oracle names list
if [ ${#ORACLE_NAMES[@]} -eq 0 ]; then
    echo "ERROR: ORACLE_NAMES list is empty"
    exit 1
fi

# Create results base directory
RESULTS_DIR="$REPO_DIR/$RESULTS_BASE_DIR"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Create temporary config directory
create_temp_config_dir

# Trap to cleanup on exit
trap cleanup_temp_configs EXIT

# Run experiments for each oracle
TOTAL_EXPERIMENTS=0

echo "=============================================="
echo "Running experiments for each TDC oracle"
echo "=============================================="
echo ""

for oracle_name in "${ORACLE_NAMES[@]}"; do
    # Get oracle-specific configuration
    direction="${ORACLE_DIRECTION[$oracle_name]}"
    
    # Create results directory for this oracle
    ORACLE_RESULTS_DIR="$RESULTS_DIR/$oracle_name"
    mkdir -p "$ORACLE_RESULTS_DIR"
    
    echo "----------------------------------------------"
    echo "Oracle: $oracle_name"
    echo "Direction: $direction"
    echo "Max iterations: $MAX_ITERATIONS"
    echo "Results directory: $ORACLE_RESULTS_DIR"
    echo "----------------------------------------------"
    
    # Run repetitions for this oracle
    for rep in $(seq 1 $NUM_REPETITIONS); do
        TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
        
        # Create experiment name
        experiment_name="${oracle_name}_${direction}_rep${rep}"
        
        # Create config file
        config_path="$TEMP_CONFIG_DIR/${experiment_name}.yaml"
        create_tdc_config \
            "$config_path" \
            "$experiment_name" \
            "$ORACLE_RESULTS_DIR" \
            "$oracle_name" \
            "$direction" \
            "$MAX_ITERATIONS"
        
        # Run experiment
        run_experiment "$config_path" "Oracle: $oracle_name, Repetition $rep / $NUM_REPETITIONS"
    done
    
    echo ""
done

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "=============================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Total experiments run: $TOTAL_EXPERIMENTS"
echo ""
echo "Results structure:"
for oracle_name in "${ORACLE_NAMES[@]}"; do
    echo "  $RESULTS_DIR/$oracle_name/ - JSON files for $oracle_name"
done
echo ""
