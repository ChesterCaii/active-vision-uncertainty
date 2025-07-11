"""
Active Vision Uncertainty Reward System

An information-theoretic uncertainty reward system for active vision 
using RGB-D images from RLBench.
"""

from .core import rgb_entropy, save_heatmap
from .visualization import create_slide_overlay, create_comparison_plot

__version__ = "0.1.0"
__all__ = ["rgb_entropy", "save_heatmap", "create_slide_overlay", "create_comparison_plot"] 