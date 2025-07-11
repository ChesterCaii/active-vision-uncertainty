"""
Unit tests for entropy computation functionality.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

from src.uncertainty.core import rgb_entropy, save_heatmap, UncertaintyModel


class TestUncertaintyModel:
    """Test cases for UncertaintyModel class."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = UncertaintyModel()
        assert model.device in ["cpu", "cuda"]
        assert model.model is not None
        assert model.input_size == 224
        assert model.patch_size == 16
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        model = UncertaintyModel()
        rgb_tensor = torch.rand(3, 224, 224)
        
        processed = model._preprocess_image(rgb_tensor)
        
        assert processed.shape == (1, 3, 224, 224)
        assert processed.device.type == model.device
    
    def test_preprocess_image_resize(self):
        """Test image preprocessing with resizing."""
        model = UncertaintyModel()
        rgb_tensor = torch.rand(3, 128, 128)
        
        processed = model._preprocess_image(rgb_tensor)
        
        assert processed.shape == (1, 3, 224, 224)
    
    def test_compute_patch_entropy(self):
        """Test patch entropy computation."""
        model = UncertaintyModel()
        
        # Create mock logits
        batch_size, num_patches, num_classes = 1, 196, 1000
        logits = torch.randn(batch_size, num_patches, num_classes)
        
        entropy = model._compute_patch_entropy(logits)
        
        assert entropy.shape == (batch_size, num_patches)
        assert torch.all(entropy >= 0)  # Entropy should be non-negative
        assert torch.all(torch.isfinite(entropy))  # Should be finite
    
    def test_reshape_to_spatial(self):
        """Test reshaping patch values to spatial grid."""
        model = UncertaintyModel()
        
        # Create patch values for 14x14 grid (196 patches)
        patch_values = torch.randn(1, 196)
        
        spatial_grid = model._reshape_to_spatial(patch_values)
        
        assert spatial_grid.shape == (1, 14, 14)


class TestRGBEntropy:
    """Test cases for rgb_entropy function."""
    
    def test_entropy_computation_basic(self):
        """Test basic entropy computation."""
        rgb_tensor = torch.rand(3, 224, 224)
        
        entropy, heatmap = rgb_entropy(rgb_tensor)
        
        # Check entropy value
        assert isinstance(entropy, float)
        assert entropy >= 0
        assert np.isfinite(entropy)
        
        # Check heatmap
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (224, 224)
        assert np.all(heatmap >= 0)
        assert np.all(np.isfinite(heatmap))
    
    def test_entropy_computation_different_sizes(self):
        """Test entropy computation with different image sizes."""
        sizes = [(224, 224), (128, 128), (256, 256)]
        
        for h, w in sizes:
            rgb_tensor = torch.rand(3, h, w)
            entropy, heatmap = rgb_entropy(rgb_tensor)
            
            assert isinstance(entropy, float)
            assert entropy >= 0
            assert heatmap.shape == (h, w)
    
    def test_entropy_deterministic(self):
        """Test that entropy computation is deterministic."""
        rgb_tensor = torch.rand(3, 224, 224)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        entropy1, heatmap1 = rgb_entropy(rgb_tensor)
        
        torch.manual_seed(42)
        entropy2, heatmap2 = rgb_entropy(rgb_tensor)
        
        assert abs(entropy1 - entropy2) < 1e-6
        assert np.allclose(heatmap1, heatmap2, atol=1e-6)
    
    def test_entropy_with_custom_model(self):
        """Test entropy computation with custom model."""
        model = UncertaintyModel()
        rgb_tensor = torch.rand(3, 224, 224)
        
        entropy, heatmap = rgb_entropy(rgb_tensor, model)
        
        assert isinstance(entropy, float)
        assert entropy >= 0
        assert heatmap.shape == (224, 224)
    
    def test_entropy_edge_cases(self):
        """Test entropy computation with edge cases."""
        # Test with zeros
        rgb_tensor = torch.zeros(3, 224, 224)
        entropy, heatmap = rgb_entropy(rgb_tensor)
        assert entropy >= 0
        assert heatmap.shape == (224, 224)
        
        # Test with ones
        rgb_tensor = torch.ones(3, 224, 224)
        entropy, heatmap = rgb_entropy(rgb_tensor)
        assert entropy >= 0
        assert heatmap.shape == (224, 224)
        
        # Test with values at boundary
        rgb_tensor = torch.rand(3, 224, 224)
        rgb_tensor = torch.clamp(rgb_tensor, 0, 1)
        entropy, heatmap = rgb_entropy(rgb_tensor)
        assert entropy >= 0
        assert heatmap.shape == (224, 224)


class TestSaveHeatmap:
    """Test cases for save_heatmap function."""
    
    def test_save_heatmap_basic(self):
        """Test basic heatmap saving."""
        # Create test data
        rgb_image = np.random.rand(224, 224, 3)
        heatmap = np.random.rand(224, 224)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_heatmap.png"
            
            save_heatmap(rgb_image, heatmap, str(output_path))
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_save_heatmap_different_sizes(self):
        """Test heatmap saving with different image sizes."""
        sizes = [(128, 128), (256, 256), (100, 150)]
        
        for h, w in sizes:
            rgb_image = np.random.rand(h, w, 3)
            heatmap = np.random.rand(h, w)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_path = Path(tmp_dir) / f"test_heatmap_{h}x{w}.png"
                
                save_heatmap(rgb_image, heatmap, str(output_path))
                
                assert output_path.exists()
    
    def test_save_heatmap_custom_params(self):
        """Test heatmap saving with custom parameters."""
        rgb_image = np.random.rand(224, 224, 3)
        heatmap = np.random.rand(224, 224)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_heatmap_custom.png"
            
            save_heatmap(
                rgb_image, 
                heatmap, 
                str(output_path),
                alpha=0.8,
                colormap="plasma"
            )
            
            assert output_path.exists()
    
    def test_save_heatmap_edge_cases(self):
        """Test heatmap saving with edge cases."""
        # Test with constant heatmap
        rgb_image = np.random.rand(224, 224, 3)
        heatmap = np.ones((224, 224)) * 0.5
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_heatmap_constant.png"
            
            save_heatmap(rgb_image, heatmap, str(output_path))
            
            assert output_path.exists()
        
        # Test with zero heatmap
        heatmap = np.zeros((224, 224))
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_heatmap_zero.png"
            
            save_heatmap(rgb_image, heatmap, str(output_path))
            
            assert output_path.exists()


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow from image to heatmap."""
        # Generate test image
        rgb_tensor = torch.rand(3, 224, 224)
        
        # Compute entropy
        entropy, heatmap = rgb_entropy(rgb_tensor)
        
        # Convert tensor to numpy for visualization
        rgb_numpy = rgb_tensor.permute(1, 2, 0).numpy()
        
        # Save heatmap
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "integration_test.png"
            
            save_heatmap(rgb_numpy, heatmap, str(output_path))
            
            assert output_path.exists()
            assert isinstance(entropy, float)
            assert entropy >= 0
            assert heatmap.shape == (224, 224)
    
    def test_multiple_images_consistency(self):
        """Test consistency across multiple images."""
        entropies = []
        heatmap_shapes = []
        
        for i in range(5):
            rgb_tensor = torch.rand(3, 224, 224)
            entropy, heatmap = rgb_entropy(rgb_tensor)
            
            entropies.append(entropy)
            heatmap_shapes.append(heatmap.shape)
        
        # Check that all entropies are valid
        assert all(isinstance(e, float) and e >= 0 for e in entropies)
        
        # Check that all heatmaps have the same shape
        assert all(shape == (224, 224) for shape in heatmap_shapes)
        
        # Check that entropies vary (not all identical)
        assert len(set(entropies)) > 1  # Should have some variation


if __name__ == "__main__":
    # Run basic tests if executed directly
    print("Running basic entropy tests...")
    
    # Test 1: Basic functionality
    test_rgb = torch.rand(3, 224, 224)
    entropy, heatmap = rgb_entropy(test_rgb)
    
    assert isinstance(entropy, float), "Entropy should be a float"
    assert entropy >= 0, "Entropy should be non-negative"
    assert np.isfinite(entropy), "Entropy should be finite"
    print(f"✓ Entropy value: {entropy:.4f}")
    
    # Test 2: Heatmap shape
    assert isinstance(heatmap, np.ndarray), "Heatmap should be numpy array"
    assert heatmap.shape == (224, 224), f"Expected (224, 224), got {heatmap.shape}"
    assert np.all(heatmap >= 0), "Heatmap values should be non-negative"
    assert np.all(np.isfinite(heatmap)), "Heatmap values should be finite"
    print(f"✓ Heatmap shape: {heatmap.shape}")
    
    # Test 3: Save heatmap
    rgb_numpy = test_rgb.permute(1, 2, 0).numpy()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test_output.png"
        save_heatmap(rgb_numpy, heatmap, str(output_path))
        assert output_path.exists(), "Heatmap file should be created"
        print(f"✓ Heatmap saved successfully")
    
    print("All basic tests passed!") 