# AI-Generated Image Detection using DINOv3 and TabPFN

Bachelor's Thesis project implementing a machine learning pipeline for detecting AI-generated images across multiple generators using DINOv3 feature extraction and TabPFN classification.

## Overview

This project evaluates the effectiveness of combining DINOv3 with TabPFN for detecting AI-generated images. The pipeline supports multiple evaluation modes to assess cross-generator generalization.

**Key Features:**
- DINOv3-based feature extraction (768-dimensional CLS token embeddings)
- PCA dimensionality reduction to 500 components
- TabPFN classification with multiple evaluation modes
- Support for multiple AI image generators (ADM, BigGAN, Midjourney, Stable Diffusion, etc.)
- Comprehensive visualization tools for results analysis

## Installation

### 1. Create Conda Environment

```bash
# Create new environment with Python 3.10
conda create -n ai_detection python=3.10 -y
conda activate ai_detection
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 3. Configure Environment Name

**Important:** Update the environment name in the SLURM scripts before running:

```bash
# Edit these files and change CONDA_ENV variable:
# - slurm/extract_dinov3.sh
# - slurm/run_tabpfn.sh

# Change this line:
CONDA_ENV="your_environment_name"  # Replace with your actual environment name
```

## Dataset Structure

Expected directory structure for the image dataset:

```
genimage_data/
├── ADM/
│   ├── train/
│   │   ├── ai/          # AI-generated images
│   │   └── nature/      # Real images
│   └── val/
│       ├── ai/
│       └── nature/
├── BigGAN/
│   └── ...
├── Midjourney/
│   └── ...
└── [other generators]/
    └── ...
```

## Usage

### 1. Feature Extraction

Extract DINOv3 CLS token features from images:

```bash
# Submit SLURM job
sbatch slurm/extract_dinov3.sh
```

**Output files** (in `features/`):
- `X_cls.npy`: Feature matrix (N × 768)
- `y.npy`: Binary labels (0=real, 1=fake)
- `gens.npy`: Generator identifiers
- `splits.npy`: Train/validation split labels
- `paths.npy`: Original image file paths

### 2. PCA Dimensionality Reduction

Reduce feature dimensions from 768 to 500 using Incremental PCA:

```bash
python scripts/run_pca_dinov3.py \
    --x_path features/X_cls.npy \
    --out_path features/X_cls_pca.npy \
    --n_components 500 \
    --batch_size 50000
```

**Output files**:
- `X_cls_pca.npy`: Reduced feature matrix (N × 500)
- `pca_model.joblib`: Fitted PCA model

### 3. TabPFN Classification

Run TabPFN evaluation across multiple modes and training sizes:

```bash
# Submit complete evaluation grid
sbatch slurm/run_tabpfn.sh
```

**Evaluation Modes:**
- `multi-multi`: Train on all generators, test on all generators
- `multi-single`: Train on all generators, test on each generator separately
- `single-multi`: Train on one generator, test on all generators
- `single-single`: Train on one generator, test on another generator

**Training Sizes:** 625, 300, 150, 75, 30, 25 samples per generator

**Results** are saved in `results_tabpfn/{mode}/tabpfn/`

### 4. Checkpoints

TabPFN checkpoints are available for download:

**📦 [Download Checkpoints](https://drive.google.com/drive/folders/1P7a4wuVAn1xLPpTYdO4TDbqy1oXHd3Ga?usp=drive_link)**

Available models:
- **Multi-generator models** (6 Checkpoints): Trained on all generators for each training size (625, 300, 150, 75, 30, 25)
- **Single-generator models** (48 Checkpoints): Trained separately on each of 8 generators (ADM, BigGAN, Glide, Midjourney, SD1.4, SD1.5, VQDM, Wukong) for each training size

**Total: 54  Checkpoints**

## Visualization

### PCA Visualization

```bash
python scripts/Plots/create_pca_combined.py \
    --base_dir features \
    --output_dir plots \
    --mode grid
```

### TabPFN Performance Plots

**Separate mode visualizations:**

```bash
# Multi-generator training → Multi-generator testing
python scripts/Plots/create_tabpfn_separate.py \
    --results_dir results_tabpfn \
    --output_dir plots \
    --mode multi-multi

# Multi-generator training → Single-generator testing
python scripts/Plots/create_tabpfn_separate.py \
    --results_dir results_tabpfn \
    --output_dir plots \
    --mode multi-single
```

**Comparison across modes:**

```bash
python scripts/Plots/create_tabpfn_comparison.py \
    --results_dir results_tabpfn \
    --output_dir plots \
    --style 1
```

### TabPFN vs LATTE Comparisons

Compare TabPFN performance against LATTE-Diffusion-Detector baseline:

```bash
# Multi-Multi comparison
python scripts/Plots/compare_tabpfn_latte.py \
    --results_dir . \
    --output plots/comparison_multi_multi.png

# Multi-Single comparison
python scripts/Plots/compare_tabpfn_latte_multi_single.py \
    --results_dir . \
    --output plots/comparison_multi_single_grid.png

# Single-Multi comparison
python scripts/Plots/compare_tabpfn_latte_single_multi.py \
    --results_dir . \
    --output plots/comparison_single_multi_grid.png \
    --exclude_sizes 25 30 75
```

### Confusion Matrices

Visualize classification performance across generators:

```bash
# Small training size (25 samples)
python scripts/Plots/create_confusion_matrices.py \
    --results_dir . \
    --output_dir plots \
    --train_size 25

# Large training size (625 samples)
python scripts/Plots/create_confusion_matrices.py \
    --results_dir . \
    --output_dir plots \
    --train_size 625
```

### Difference Heatmap

Analyze performance differences across generator pairs:

```bash
python scripts/Plots/create_difference_heatmap.py \
    --results_dir . \
    --output_dir plots \
    --train_size 625
```

## Project Structure

```
.
├── genimage_data/           # Image dataset (not included in repository)
├── features/
│   └── DINOv3/             # Extracted features and PCA-reduced features
├── results_tabpfn/         # TabPFN evaluation results (JSON files)
├── plots/                  # Generated visualizations
├── scripts/
│   ├── extract_dinov3.py   # Feature extraction script
│   ├── run_pca_dinov3.py   # PCA dimensionality reduction
│   ├── eval_tabpfn.py      # TabPFN evaluation
│   └── Plots/              # Visualization scripts
│       ├── create_pca_combined.py
│       ├── create_tabpfn_separate.py
│       ├── create_tabpfn_comparison.py
│       ├── compare_tabpfn_latte.py
│       ├── compare_tabpfn_latte_multi_single.py
│       ├── compare_tabpfn_latte_single_multi.py
│       ├── create_confusion_matrices.py
│       └── create_difference_heatmap.py
├── slurm/
│   ├── extract_dinov3.sh   # SLURM job for feature extraction
│   └── run_tabpfn.sh       # SLURM job for TabPFN evaluation grid
├── requirements.txt        # Python dependencies
└── README.md
```

## Results

Results are organized as follows:

```
results_tabpfn/
├── multi-multi/
│   └── tabpfn/
│       ├── multi-multi_625_tabpfn.json
│       ├── multi-multi_300_tabpfn.json
│       └── ...
├── multi-single/
│   └── tabpfn/
│       ├── multi-single_625_tabpfn_adm.json
│       ├── multi-single_625_tabpfn_biggan.json
│       └── ...
├── single-multi/
│   └── tabpfn/
│       └── ...
└── single-single/
    └── tabpfn/
        └── ...
```

Each JSON file contains:
```json
{
    "mode": "multi-multi",
    "train_size": 625,
    "train_gen": null,
    "test_gen": null,
    "seed": 42,
    "metrics": {
        "accuracy": 0.8765,
        "precision": 0.865604971839192,
        "recall": 0.8914,
        "f1": 0.8783131342989458,
        "roc_auc": 0.9501013599999999,
        "n_test": 10000,
    }
}
```