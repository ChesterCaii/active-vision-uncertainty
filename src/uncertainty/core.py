"""
Core entropy computation module for uncertainty-based active vision.
"""

import torch
import torch.nn.functional as F
import numpy as np
import timm
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


class UncertaintyModel:
    """Vision Transformer-based uncertainty model."""
    
    def __init__(self, model_name: str = "vit_base_patch16_224", device: Optional[str] = None):
        """
        Initialize the uncertainty model.
        
        Args:
            model_name: Name of the timm model to use
            device: Device to run inference on (auto-detects if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained Vision Transformer
        self.model = timm.create_model(model_name, pretrained=True, num_classes=1000)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get model configuration
        self.input_size = self.model.default_cfg['input_size'][-1]  # Assuming square input
        self.patch_size = 16  # Standard for base model
        
    def _preprocess_image(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Preprocess RGB tensor for model input.
        
        Args:
            rgb_tensor: RGB tensor of shape (3, H, W) with values in [0, 1]
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Ensure tensor is on correct device
        rgb_tensor = rgb_tensor.to(self.device)
        
        # Resize to model input size
        if rgb_tensor.shape[-1] != self.input_size:
            rgb_tensor = F.interpolate(
                rgb_tensor.unsqueeze(0), 
                size=(self.input_size, self.input_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # Normalize using ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(3, 1, 1)
        normalized = (rgb_tensor - mean) / std
        
        return normalized.unsqueeze(0)  # Add batch dimension
    
    def _compute_patch_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute Shannon entropy for each patch.
        
        Args:
            logits: Model logits of shape (batch, num_patches, num_classes)
            
        Returns:
            Entropy values for each patch
        """
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Compute Shannon entropy: H = -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)  # Add small epsilon for numerical stability
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        return entropy
    
    def _reshape_to_spatial(self, patch_values: torch.Tensor) -> torch.Tensor:
        """
        Reshape patch values to spatial grid.
        
        Args:
            patch_values: Values for each patch (batch, num_patches)
            
        Returns:
            Spatial grid of values
        """
        batch_size, num_patches = patch_values.shape
        grid_size = int(np.sqrt(num_patches))
        
        return patch_values.view(batch_size, grid_size, grid_size)


def rgb_entropy(rgb_tensor: torch.Tensor, model: Optional[UncertaintyModel] = None) -> Tuple[float, np.ndarray]:
    """
    Compute Shannon entropy from RGB tensor using Vision Transformer features.
    
    Args:
        rgb_tensor: RGB tensor of shape (3, H, W) with values in [0, 1]
        model: Optional pre-initialized uncertainty model
        
    Returns:
        Tuple of (scalar_entropy, HxW_heatmap_numpy)
    """
    if model is None:
        model = UncertaintyModel()
    
    with torch.no_grad():
        # Preprocess input
        input_tensor = model._preprocess_image(rgb_tensor)
        
        # Get model features (we need to modify the forward pass)
        # For ViT, we extract patch embeddings before classification
        x = model.model.patch_embed(input_tensor)
        cls_token = model.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = model.model.pos_drop(x + model.model.pos_embed)
        
        # Pass through transformer blocks
        for block in model.model.blocks:
            x = block(x)
        
        x = model.model.norm(x)
        
        # Remove CLS token and get patch embeddings
        patch_embeddings = x[:, 1:, :]  # Skip CLS token
        
        # Apply classification head to each patch
        patch_logits = model.model.head(patch_embeddings)
        
        # Compute entropy for each patch
        patch_entropies = model._compute_patch_entropy(patch_logits)
        
        # Reshape to spatial grid
        entropy_grid = model._reshape_to_spatial(patch_entropies)
        
        # Convert to numpy and resize to original image size
        entropy_map = entropy_grid.squeeze().cpu().numpy()
        
        # Resize entropy map to match original image size
        from scipy.ndimage import zoom
        h, w = rgb_tensor.shape[1], rgb_tensor.shape[2]
        zoom_factors = (h / entropy_map.shape[0], w / entropy_map.shape[1])
        entropy_heatmap = zoom(entropy_map, zoom_factors, order=1)
        
        # Compute scalar entropy (mean of all patches)
        scalar_entropy = float(torch.mean(patch_entropies).item())
    
    return scalar_entropy, entropy_heatmap


def save_heatmap(
    rgb_image: np.ndarray,
    heatmap: np.ndarray,
    output_path: str,
    alpha: float = 0.6,
    colormap: str = "viridis"
) -> None:
    """
    Save uncertainty heatmap overlaid on RGB image.
    
    Args:
        rgb_image: RGB image as numpy array (H, W, 3) with values in [0, 1]
        heatmap: Uncertainty heatmap as numpy array (H, W)
        output_path: Path to save the visualization
        alpha: Transparency of the heatmap overlay
        colormap: Matplotlib colormap name
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display base image
    ax.imshow(rgb_image)
    
    # Overlay heatmap
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap_colored = cm.get_cmap(colormap)(heatmap_normalized)
    ax.imshow(heatmap_colored, alpha=alpha)
    
    # Add colorbar
    mappable = cm.ScalarMappable(cmap=colormap)
    mappable.set_array(heatmap)
    cbar = plt.colorbar(mappable, ax=ax)
    cbar.set_label("Uncertainty (Shannon Entropy)", rotation=270, labelpad=20)
    
    # Set title and remove axis ticks
    ax.set_title("Uncertainty Heatmap Overlay")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to: {output_path}")


# Global model instance for efficiency
_global_model = None


def get_global_model() -> UncertaintyModel:
    """Get or create global model instance."""
    global _global_model
    if _global_model is None:
        _global_model = UncertaintyModel()
    return _global_model


if __name__ == "__main__":
    # Simple test
    print("Testing entropy computation...")
    test_rgb = torch.rand(3, 224, 224)
    entropy, heatmap = rgb_entropy(test_rgb)
    print(f"Test entropy: {entropy:.4f}")
    print(f"Heatmap shape: {heatmap.shape}")
    print("Test completed successfully!") 