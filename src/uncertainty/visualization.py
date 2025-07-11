"""
Enhanced visualization module for slide-quality uncertainty overlays.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from scipy.ndimage import zoom
from typing import Optional, Tuple
from pathlib import Path


def create_slide_overlay(
    rgb_image: np.ndarray,
    entropy_map: np.ndarray,
    output_path: str,
    alpha: float = 0.6,
    colormap: str = "viridis",
    dpi: int = 300,
    title: str = "Uncertainty Heatmap Overlay",
    show_colorbar: bool = True,
    figsize: Tuple[float, float] = (10, 8)
) -> None:
    """
    Create a high-quality uncertainty overlay for slide presentations.
    
    Args:
        rgb_image: RGB image as numpy array (H, W, 3) with values in [0, 1]
        entropy_map: Uncertainty map as numpy array (H, W)
        output_path: Path to save the visualization
        alpha: Transparency of the heatmap overlay (0-1)
        colormap: Matplotlib colormap name
        dpi: Resolution for saving (300 for slides)
        title: Title for the plot
        show_colorbar: Whether to show the colorbar
        figsize: Figure size in inches
    """
    # Ensure entropy map matches RGB image size with high-quality upsampling
    if entropy_map.shape != rgb_image.shape[:2]:
        h, w = rgb_image.shape[:2]
        zoom_factors = (h / entropy_map.shape[0], w / entropy_map.shape[1])
        # Use high-quality interpolation for upsampling
        entropy_map = zoom(entropy_map, zoom_factors, order=3, mode='reflect')
    
    # Create figure with high quality settings
    plt.style.use('default')  # Clean style
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Display base RGB image
    ax.imshow(rgb_image, aspect='equal')
    
    # Create normalized heatmap with better contrast
    entropy_normalized = normalize_for_display(entropy_map)
    
    # Create overlay with specified colormap
    overlay = ax.imshow(
        entropy_normalized, 
        cmap=colormap, 
        alpha=alpha,
        aspect='equal',
        interpolation='bilinear'  # Smooth interpolation
    )
    
    # Configure plot appearance
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add colorbar if requested
    if show_colorbar:
        cbar = plt.colorbar(overlay, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label("Uncertainty (Shannon Entropy)", rotation=270, labelpad=20, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
    
    # Save with high quality
    plt.tight_layout()
    plt.savefig(
        output_path, 
        dpi=dpi, 
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        format='png'
    )
    plt.close()
    
    print(f"High-quality overlay saved to: {output_path}")


def normalize_for_display(entropy_map: np.ndarray) -> np.ndarray:
    """
    Normalize entropy map for optimal display contrast.
    
    Args:
        entropy_map: Raw entropy values
        
    Returns:
        Normalized entropy map with enhanced contrast
    """
    # Handle edge cases
    if np.all(entropy_map == entropy_map.flat[0]):
        # Constant map - return zeros
        return np.zeros_like(entropy_map)
    
    # Apply percentile-based normalization for better contrast
    p5, p95 = np.percentile(entropy_map, [5, 95])
    normalized = np.clip((entropy_map - p5) / (p95 - p5), 0, 1)
    
    # Apply gamma correction for better visual perception
    gamma = 0.8
    normalized = np.power(normalized, gamma)
    
    return normalized


def create_comparison_plot(
    rgb_images: list,
    entropy_maps: list,
    labels: list,
    output_path: str,
    alpha: float = 0.6,
    colormap: str = "viridis",
    dpi: int = 300,
    figsize: Tuple[float, float] = (15, 5)
) -> None:
    """
    Create a comparison plot showing multiple viewpoints and their uncertainty maps.
    
    Args:
        rgb_images: List of RGB images
        entropy_maps: List of corresponding entropy maps
        labels: List of labels for each image
        output_path: Path to save the comparison
        alpha: Transparency for overlays
        colormap: Colormap to use
        dpi: Resolution for saving
        figsize: Figure size
    """
    n_images = len(rgb_images)
    fig, axes = plt.subplots(2, n_images, figsize=figsize, facecolor='white')
    
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_images):
        # Top row: original images
        axes[0, i].imshow(rgb_images[i])
        axes[0, i].set_title(f"{labels[i]} - Original", fontsize=12, fontweight='bold')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Bottom row: uncertainty overlays
        axes[1, i].imshow(rgb_images[i])
        
        # Upsample entropy map if needed
        entropy_map = entropy_maps[i]
        if entropy_map.shape != rgb_images[i].shape[:2]:
            h, w = rgb_images[i].shape[:2]
            zoom_factors = (h / entropy_map.shape[0], w / entropy_map.shape[1])
            entropy_map = zoom(entropy_map, zoom_factors, order=3, mode='reflect')
        
        normalized_entropy = normalize_for_display(entropy_map)
        overlay = axes[1, i].imshow(normalized_entropy, cmap=colormap, alpha=alpha)
        
        axes[1, i].set_title(f"{labels[i]} - Uncertainty", fontsize=12, fontweight='bold')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    
    # Remove spines for cleaner look
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Add a single colorbar for all plots
    cbar = fig.colorbar(overlay, ax=axes, shrink=0.6, aspect=40, pad=0.02)
    cbar.set_label("Uncertainty (Shannon Entropy)", rotation=270, labelpad=15, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        format='png'
    )
    plt.close()
    
    print(f"Comparison plot saved to: {output_path}")


def create_slide_demo() -> None:
    """Create demonstration visualizations optimized for slides."""
    import sys
    import numpy as np
    sys.path.append('src')
    from uncertainty.core import rgb_entropy
    from uncertainty.scan_views import generate_synthetic_image
    
    print("🎨 Creating slide-quality visualizations...")
    
    # Create slides directory
    slides_dir = Path("slides")
    slides_dir.mkdir(exist_ok=True)
    
    # Generate different viewpoints for comparison
    poses_and_labels = [
        (np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), "Front View"),
        (np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]), "Side View"),
        (np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]), "Top View")
    ]
    
    rgb_images = []
    entropy_maps = []
    labels = []
    
    for pose, label in poses_and_labels:
        # Generate synthetic image
        synthetic_img = generate_synthetic_image(pose)
        
        # Compute entropy
        entropy, heatmap = rgb_entropy(synthetic_img)
        
        # Convert to numpy for visualization
        img_numpy = synthetic_img.permute(1, 2, 0).numpy()
        
        # Store for comparison plot
        rgb_images.append(img_numpy)
        entropy_maps.append(heatmap)
        labels.append(label)
        
        # Create individual high-quality overlay
        create_slide_overlay(
            img_numpy, 
            heatmap, 
            f"slides/{label.lower().replace(' ', '_')}_entropy_overlay.png",
            title=f"Uncertainty Analysis - {label}",
            alpha=0.6
        )
        
        print(f"✅ Created {label} overlay (entropy: {entropy:.4f})")
    
    # Create comparison plot
    create_comparison_plot(
        rgb_images,
        entropy_maps,
        labels,
        "slides/viewpoint_comparison.png",
        figsize=(15, 8)
    )
    
    print("🎉 Slide-quality visualizations complete!")
    print("📁 Check the slides/ directory for high-resolution images")


if __name__ == "__main__":
    create_slide_demo() 