"""
Active Vision Uncertainty Reward System

An information-theoretic uncertainty reward system for active vision 
using RGB-D images from RLBench.
"""

from .core import rgb_entropy, save_heatmap

__version__ = "0.1.0"
__all__ = ["rgb_entropy", "save_heatmap"] 