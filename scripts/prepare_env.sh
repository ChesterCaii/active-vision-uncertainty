#!/bin/bash

# Active Vision Uncertainty Reward Environment Setup Script
# This script creates a conda environment and installs all required dependencies

set -e  # Exit on any error

echo "Setting up Active Vision Uncertainty Reward environment..."

# Environment name
ENV_NAME="active-vision-uncertainty"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove it and create a new one? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "Skipping environment creation."
        exit 0
    fi
fi

# Create new conda environment
echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.9 -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (with CUDA support if available)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install requirements from requirements.txt
echo "Installing project dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Installing dependencies manually..."
    pip install timm>=0.9
    pip install numpy
    pip install matplotlib
    pip install open3d
    pip install pytest
    pip install tqdm
    pip install jupyter
    pip install pandas
    pip install scipy
    pip install Pillow
    
    # Try to install RLBench (may fail if dependencies are missing)
    echo "Attempting to install RLBench..."
    pip install rlbench || echo "Warning: RLBench installation failed. This is normal if you don't have CoppeliaSim installed."
fi

# Install project in development mode
echo "Installing project in development mode..."
pip install -e .

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/raw
mkdir -p results
mkdir -p logs

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import timm; print(f'timm version: {timm.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Test basic functionality
echo "Testing basic functionality..."
python -c "
import sys
sys.path.append('src')
from uncertainty.core import rgb_entropy
import torch
print('Testing entropy computation...')
test_rgb = torch.rand(3, 224, 224)
entropy, heatmap = rgb_entropy(test_rgb)
print(f'Test successful! Entropy: {entropy:.4f}, Heatmap shape: {heatmap.shape}')
"

echo ""
echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To test the installation, run:"
echo "  python -m pytest tests/"
echo ""
echo "To run the demo notebook:"
echo "  jupyter notebook notebooks/sanity_checks.ipynb"
echo ""
echo "To capture data from RLBench:"
echo "  python -m uncertainty.rlbench_runner --n 30 --out data/raw"
echo ""
echo "To scan viewpoints:"
echo "  python -m uncertainty.scan_views --n 180 --out results/grid.csv" 