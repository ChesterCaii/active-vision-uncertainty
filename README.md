# Active Vision Uncertainty Reward

An information-theoretic uncertainty reward system for active vision using RGB-D images from RLBench. This project implements entropy-based uncertainty measures to guide robotic vision systems in selecting optimal viewpoints.

## Project Overview

This repository implements an uncertainty-based reward system for active vision tasks. The system:
- Captures RGB-D images from RLBench simulation environment
- Computes information-theoretic uncertainty using Vision Transformer features
- Generates uncertainty heatmaps for visual analysis
- Provides systematic viewpoint scanning capabilities

## Quick Start

1. **Setup Environment**
   ```bash
   chmod +x scripts/prepare_env.sh
   ./scripts/prepare_env.sh
   conda activate active-vision-uncertainty
   ```

2. **Run Tests**
   ```bash
   pytest tests/
   ```

3. **Generate Sample Data**
   ```bash
   python -m uncertainty.rlbench_runner --n 30 --out data/raw
   ```

4. **Scan Viewpoints**
   ```bash
   python -m uncertainty.scan_views --n 180 --out results/grid.csv
   ```

5. **Demo Notebook**
   ```bash
   jupyter notebook notebooks/sanity_checks.ipynb
   ```

## Repository Structure

```
active-vision-uncertainty/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── uncertainty/
│       ├── __init__.py
│       ├── core.py              # Core entropy computation
│       ├── rlbench_runner.py    # RLBench data capture
│       └── scan_views.py        # Systematic viewpoint scanning
├── tests/
│   └── test_entropy.py
├── notebooks/
│   └── sanity_checks.ipynb
├── scripts/
│   └── prepare_env.sh
└── slides/
    └── placeholder.txt
```

## Key Components

### Core Entropy Module (`src/uncertainty/core.py`)
- `rgb_entropy()`: Computes Shannon entropy from Vision Transformer features
- `save_heatmap()`: Visualizes uncertainty overlays on images

### RLBench Runner (`src/uncertainty/rlbench_runner.py`)
- Captures RGB-D images with randomized camera poses
- Saves data in NPZ format with RGB, depth, and pose information

### Viewpoint Scanner (`src/uncertainty/scan_views.py`)
- Systematic sphere-based viewpoint sampling
- Exports entropy measurements to CSV format

## Milestone Checklist

- [ ] Environment setup and dependencies
- [ ] Core entropy computation implementation
- [ ] RLBench integration and data capture
- [ ] Systematic viewpoint scanning
- [ ] Visualization and heatmap generation
- [ ] Unit tests and validation
- [ ] Demo notebook and examples
- [ ] Documentation and presentation materials
- [ ] Performance optimization for single GPU
- [ ] Final validation and testing

## Requirements

- Python 3.8+
- PyTorch 2.3+
- RLBench 1.3+
- Single GPU (laptop compatible)

## License

MIT License 