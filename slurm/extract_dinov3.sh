#!/bin/bash
#SBATCH --job-name=dinov3_extract
#SBATCH --partition=gpu_a100_short 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=127500mb
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --chdir=/pfs/work9/workspace/scratch/ma_jwaltea-genimage_data/Bachelors-Thesis-backup-20251214_222745
#SBATCH --output=logs/extract_features_%j.out
#SBATCH --error=logs/extract_features_%j.err

################################################################################
# DINOv3 Feature Extraction
# Extracts CLS token features from image dataset
################################################################################

set -e  # Exit on error

echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "========================================="

# Configuration
CONDA_ENV="your_environment_name"  # Configure your environment name here

# Relative paths (we're already in project root thanks to --chdir)
IMAGES_ROOT="genimage_data"
OUTPUT_DIR="features/"

# Create output directories
mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

echo "Images root: ${PWD}/${IMAGES_ROOT}"
echo "Output directory: ${PWD}/${OUTPUT_DIR}"

# Load modules - skip if not available
module purge 2>/dev/null || true
module load compiler/gnu devel/cuda 2>/dev/null || echo "[INFO] Modules not loaded (not critical)"

# Activate conda environment
eval "$(conda shell.bash hook)" || { echo "[FATAL] conda hook failed"; exit 99; }
conda activate ${CONDA_ENV} || { echo "[FATAL] conda activate failed"; exit 98; }

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Verify GPU availability
echo ""
echo "Checking GPU..."
nvidia-smi || echo "[WARN] No GPU visible"

# Set environment variables
export MKL_THREADING_LAYER=GNU
export CUDA_VISIBLE_DEVICES=0

# Verify files exist
if [ ! -f "scripts/extract_dinov3.py" ]; then
    echo "[ERROR] extract_dinov3.py not found in $(pwd)"
    ls -la
    exit 1
fi

if [ ! -d "${IMAGES_ROOT}" ]; then
    echo "[ERROR] Images directory not found: ${PWD}/${IMAGES_ROOT}"
    exit 1
fi

# Run feature extraction
echo ""
echo "========================================="
echo "Starting feature extraction..."
echo "========================================="

python scripts/extract_dinov3.py \
    --images_root "${IMAGES_ROOT}" \
    --out_dir "${OUTPUT_DIR}" \
    --batch_size 32 \
    --hf_model facebook/dinov3-vitb16-pretrain-lvd1689m

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Feature extraction finished with exit code: $EXIT_CODE"
echo "Job ended at: $(date)"
echo "========================================="

# Verify output files
if [ -f "${OUTPUT_DIR}/X_cls.npy" ]; then
    echo "✓ X_cls.npy created successfully"
    ls -lh "${OUTPUT_DIR}/"
else
    echo "✗ ERROR: X_cls.npy not found!"
    echo "Output directory contents:"
    ls -la "${OUTPUT_DIR}/" 2>/dev/null || echo "Directory does not exist"
    exit 1
fi

exit $EXIT_CODE