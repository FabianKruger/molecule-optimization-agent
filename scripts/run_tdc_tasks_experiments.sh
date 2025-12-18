#!/bin/zsh
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
    "isomers_c11h24"
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

# Direction for all oracles (all TDC tasks are maximize - higher is better)
DIRECTION="maximize"

# Maximum number of iterations per experiment
MAX_ITERATIONS=51

# Number of repetitions per oracle
NUM_REPETITIONS=1

# LLM model name
LLM_MODEL="claude-opus-4.5"

# LLM temperature
TEMPERATURE=0.0

# Recursion limit for LangGraph
RECURSION_LIMIT=300

# Results base directory (will be created if it doesn't exist)
RESULTS_BASE_DIR="results/tdc_tasks"

# ==============================================================================
# PATHS - Relative to molecule-optimization-agent directory
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
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
# Args: $1=output_path, $2=experiment_name, $3=log_dir, $4=oracle_name
create_tdc_config() {
    local output_path="$1"
    local experiment_name="$2"
    local log_dir="$3"
    local oracle_name="$4"

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
    direction: ${DIRECTION}
    max_iterations: ${MAX_ITERATIONS}
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
echo "  Direction: $DIRECTION"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Repetitions per oracle: $NUM_REPETITIONS"
echo "  LLM model: $LLM_MODEL"
echo "  Temperature: $TEMPERATURE"
echo "  Results base directory: $RESULTS_BASE_DIR"
echo ""

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
    # Create results directory for this oracle
    ORACLE_RESULTS_DIR="$RESULTS_DIR/$oracle_name"
    mkdir -p "$ORACLE_RESULTS_DIR"
    
    echo "----------------------------------------------"
    echo "Oracle: $oracle_name"
    echo "Results directory: $ORACLE_RESULTS_DIR"
    echo "----------------------------------------------"
    
    # Run repetitions for this oracle
    for rep in $(seq 1 $NUM_REPETITIONS); do
        TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
        
        # Create experiment name
        experiment_name="${oracle_name}_rep${rep}"
        
        # Create config file
        config_path="$TEMP_CONFIG_DIR/${experiment_name}.yaml"
        create_tdc_config "$config_path" "$experiment_name" "$ORACLE_RESULTS_DIR" "$oracle_name"
        
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
