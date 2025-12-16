#!/bin/bash
#SBATCH --job-name=tabpfn_eval
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=127500mb
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --output=logs/tabpfn_eval_%j.out
#SBATCH --error=logs/tabpfn_eval_%j.err
#SBATCH --chdir=/pfs/work9/workspace/scratch/ma_jwaltea-genimage_data/Bachelors-Thesis-backup-20251214_222745

################################################################################
# TabPFN Classifier Evaluation - Grid Search
#
# Evaluates TabPFN classifier across multiple training modes, training set
# sizes, and generator combinations. Implements a systematic grid search
# over all experimental configurations.
################################################################################

set -u
set -o pipefail

# Environment Configuration

# Intel MKL threading configuration
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# TabPFN configuration
export TABPFN_ALLOW_CPU_LARGE_DATASET=1
  
# Conda Environment Setup

CONDA_ENV="your_environment_name"  # Configure your environment name here

echo "[INFO] Activating conda environment: ${CONDA_ENV}"
eval "$(conda shell.bash hook)" || { 
    echo "[FATAL] Failed to initialize conda" 
    exit 99 
}
conda activate ${CONDA_ENV} || { 
    echo "[FATAL] Failed to activate environment: ${CONDA_ENV}" 
    exit 98 
}


# GPU Verification

echo "[INFO] Verifying GPU availability..."
nvidia-smi || echo "[WARNING] No GPU detected - will use CPU"

# Experimental Configuration

# Evaluation modes to test
EVALUATION_MODES=(
    "multi-multi"    # Train: all generators, Test: all generators
    "multi-single"   # Train: all generators, Test: single generator
    "single-multi"   # Train: single generator, Test: all generators
    "single-single"  # Train: single generator, Test: single generator
)

# Training set sizes (samples per generator)
TRAINING_SIZES=(625 300 150 75 30 25)

echo "============================================================"
echo "TABPFN EVALUATION GRID SEARCH"
echo "============================================================"
echo "Start time: $(date)"
echo "Evaluation modes: ${EVALUATION_MODES[*]}"
echo "Training sizes: ${TRAINING_SIZES[*]}"
echo "============================================================"

# Detect Available Generators

echo ""
echo "[INFO] Detecting available generators from dataset..."

GENERATORS=$(python - << 'EOF'
import numpy as np
from pathlib import Path

feature_dir = Path("features")
generators = np.load(feature_dir / "gens.npy", allow_pickle=True)
labels = np.load(feature_dir / "y.npy", allow_pickle=True)

# Extract unique fake generators
fake_generators = sorted(
    set(gen for gen, label in zip(generators, labels) if label == 1)
)
print(" ".join(fake_generators))
EOF
)

echo "[INFO] Available generators: ${GENERATORS}"
echo "------------------------------------------------------------"

# Grid Search Execution

for mode in "${EVALUATION_MODES[@]}"; do
    for train_size in "${TRAINING_SIZES[@]}"; do

        echo ""
        echo "###################################################"
        echo "CONFIGURATION: mode=${mode} | train_size=${train_size}"
        echo "###################################################"

        
        # Mode 1: Multi-generator training, Multi-generator testing
        
        if [[ "$mode" == "multi-multi" ]]; then
            python -u scripts/eval_tabpfn.py \
                --mode "$mode" \
                --train_size "$train_size" \
                --device "auto" \
                --configs 32 \
                --save_schema "full" \
                --seed 42
            continue
        fi

        
        # Mode 2: Multi-generator training, Single-generator testing
        
        if [[ "$mode" == "multi-single" ]]; then
            for test_gen in $GENERATORS; do
                echo "[MODE: multi-single] Testing on generator: ${test_gen}"
                python -u scripts/eval_tabpfn.py \
                    --mode "$mode" \
                    --train_size "$train_size" \
                    --test_gen "$test_gen" \
                    --device "auto" \
                    --configs 32 \
                    --save_schema "full" \
                    --seed 42
            done
            continue
        fi

        
        # Mode 3: Single-generator training, Multi-generator testing
        
        if [[ "$mode" == "single-multi" ]]; then
            for train_gen in $GENERATORS; do
                echo "[MODE: single-multi] Training on generator: ${train_gen}"
                python -u scripts/eval_tabpfn.py \
                    --mode "$mode" \
                    --train_size "$train_size" \
                    --train_gen "$train_gen" \
                    --device "auto" \
                    --configs 32 \
                    --save_schema "full" \
                    --seed 42
            done
            continue
        fi

       
        # Mode 4: Single-generator training, Single-generator testing
      
        if [[ "$mode" == "single-single" ]]; then
            for train_gen in $GENERATORS; do
                for test_gen in $GENERATORS; do
                    echo "[MODE: single-single] train=${train_gen} | test=${test_gen}"
                    python -u scripts/eval_tabpfn.py \
                        --mode "$mode" \
                        --train_size "$train_size" \
                        --train_gen "$train_gen" \
                        --test_gen "$test_gen" \
                        --device "auto" \
                        --configs 32 \
                        --save_schema "full" \
                        --seed 42
                done
            done
            continue
        fi

    done
done

# Completion Summary

echo ""
echo "============================================================"
echo "TABPFN EVALUATION GRID SEARCH COMPLETED"
echo "============================================================"
echo "End time: $(date)"
echo "Results saved in: results_tabpfn/"
echo "============================================================"