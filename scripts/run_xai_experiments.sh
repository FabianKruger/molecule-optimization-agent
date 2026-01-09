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
random_molecule="C1CCC(CCCCC(C1)NC2=NCCN2)NC3=NCCN3"  # Example: Contains cyclodecane (https://pubchem.ncbi.nlm.nih.gov/compound/172054415)

# Similarity + QED objective thresholds
sim_threshold=0.7          # min_similarity
qed_threshold_sim=0.7      # min_qed for similarity_qed
target_score=0.8          # target combined score

# IC50 objectives thresholds
ic50_threshold=50.0        # target_ic50_nM
qed_threshold_ic50=0.5     # min_qed for ic50_qed_novel

# General experiment settings
max_iterations=50
llm_model="gpt-5.1"
temperature=0.3
recursion_limit=300

# Number of repetitions per experiment
num_repetitions=5

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

# Create an ic50mpro_novel config file
# Args: $1=output_path, $2=experiment_name, $3=log_dir, $4=xai_mode
create_ic50mpro_novel_config() {
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
          model_name: "data/xgboost/xgb_maccs_best"
      - name: novel
        params: {}
    weights: [1.0, 0.0]
    names: ["IC50", "Novelty"]

objective:
  name: ic50mpro_novel
  params:
    target_ic50_nM: ${ic50_threshold}
    max_iterations: ${max_iterations}
    xai: ${xai_mode}
EOF
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
          model_name: "data/xgboost/xgb_maccs_best"
      - name: qed
        params: {}
      - name: novel
        params: {}
    weights: [1.0, 0.0, 0.0]
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
echo "  Relevant molecule: $relevant_molecule"
echo "  Random molecule: $random_molecule"
echo "  Similarity threshold: $sim_threshold"
echo "  QED threshold (sim): $qed_threshold_sim"
echo "  Target score: $target_score"
echo "  IC50 threshold: $ic50_threshold"
echo "  QED threshold (IC50): $qed_threshold_ic50"
echo "  Max iterations: $max_iterations"
echo "  LLM model: $llm_model"
echo "  Temperature: $temperature"
echo "  Repetitions: $num_repetitions"
echo ""

# Create temporary config directory
create_temp_config_dir

# Trap to cleanup on exit
trap cleanup_temp_configs EXIT

# ==============================================================================
# EXPERIMENT 1: similarity_qed with relevant_molecule
# ==============================================================================

: '
echo "=============================================="
echo "EXPERIMENT 1: similarity_qed - relevant_molecule"
echo "=============================================="

for xai_mode in "full" "partial" "none"; do
    xai_dir_name="${xai_mode}_xai"
    if [ "$xai_mode" == "none" ]; then
        xai_dir_name="no_xai"
    fi
    
    log_dir="$RESULTS_BASE_DIR/similarity_qed/relevant_molecule/$xai_dir_name"
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
echo "EXPERIMENT 2: similarity_qed - random_molecule"
echo "=============================================="

for xai_mode in "full" "partial" "none"; do
    xai_dir_name="${xai_mode}_xai"
    if [ "$xai_mode" == "none" ]; then
        xai_dir_name="no_xai"
    fi
    
    log_dir="$RESULTS_BASE_DIR/similarity_qed/random_molecule/$xai_dir_name"
    mkdir -p "$log_dir"
    
    echo ""
    echo "XAI mode: $xai_mode ($xai_dir_name)"
    
    for rep in $(seq 1 $num_repetitions); do
        config_path="$TEMP_CONFIG_DIR/sim_qed_random_${xai_mode}_${rep}.yaml"
        experiment_name="sim_qed_random_${xai_mode}_rep${rep}"
        
        create_similarity_qed_config "$config_path" "$experiment_name" "$log_dir" "$random_molecule" "$xai_mode"
        run_experiment "$config_path" "Repetition $rep / $num_repetitions"
    done
done
'

echo ""
echo "=============================================="
echo "EXPERIMENT 3: ic50mpro_novel"
echo "=============================================="

for xai_mode in "full" "none"; do
    xai_dir_name="${xai_mode}_xai"
    if [ "$xai_mode" == "none" ]; then
        xai_dir_name="no_xai"
    fi
    
    log_dir="$RESULTS_BASE_DIR/ic50mpro/$xai_dir_name"
    mkdir -p "$log_dir"
    
    echo ""
    echo "XAI mode: $xai_mode ($xai_dir_name)"
    
    for rep in $(seq 1 $num_repetitions); do
        config_path="$TEMP_CONFIG_DIR/ic50mpro_novel_${xai_mode}_${rep}.yaml"
        experiment_name="ic50mpro_novel_${xai_mode}_rep${rep}"
        
        create_ic50mpro_novel_config "$config_path" "$experiment_name" "$log_dir" "$xai_mode"
        run_experiment "$config_path" "Repetition $rep / $num_repetitions"
    done
done

echo ""
echo "=============================================="
echo "EXPERIMENT 4: ic50mpro_qed_novel"
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
echo ""
echo "Directory structure:"
echo "  similarity_qed/"
echo "    relevant_molecule/{full_xai, partial_xai, no_xai}"
echo "    random_molecule/{full_xai, partial_xai, no_xai}"
echo "  ic50mpro/{full_xai, no_xai}"
echo "  ic50mpro_qed/{full_xai, partial_xai, no_xai}"
echo ""
echo "Total experiments run: $((6 * num_repetitions + 2 * num_repetitions + 3 * num_repetitions))"
