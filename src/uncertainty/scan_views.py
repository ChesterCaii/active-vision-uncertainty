"""
Systematic viewpoint scanning module for uncertainty analysis.
"""

import argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
import math
from tqdm import tqdm

from .core import rgb_entropy, get_global_model


def generate_sphere_viewpoints(
    num_points: int,
    radius: float = 1.0,
    center: np.ndarray = None,
    elevation_range: Tuple[float, float] = (10.0, 170.0)
) -> List[np.ndarray]:
    """
    Generate viewpoints on a sphere around a target.
    
    Args:
        num_points: Number of viewpoints to generate
        radius: Radius of the sphere
        center: Center point of the sphere (default: origin)
        elevation_range: Range of elevation angles in degrees (theta from z-axis)
        
    Returns:
        List of viewpoint poses [x, y, z, qw, qx, qy, qz]
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    
    viewpoints = []
    
    # Convert elevation range to radians
    theta_min, theta_max = np.radians(elevation_range)
    
    # Generate points using Fibonacci spiral for uniform distribution
    golden_ratio = (1 + 5**0.5) / 2
    
    for i in range(num_points):
        # Normalized index
        norm_i = i / (num_points - 1) if num_points > 1 else 0
        
        # Spherical coordinates
        theta = theta_min + (theta_max - theta_min) * norm_i  # Elevation
        phi = 2 * math.pi * i / golden_ratio  # Azimuth
        
        # Convert to Cartesian coordinates
        x = radius * math.sin(theta) * math.cos(phi)
        y = radius * math.sin(theta) * math.sin(phi)
        z = radius * math.cos(theta)
        
        # Translate to center
        position = center + np.array([x, y, z])
        
        # Compute orientation (looking at center)
        direction = center - position
        direction = direction / np.linalg.norm(direction)
        
        # Create quaternion for look-at orientation
        # Simplified: assume up vector is [0, 0, 1]
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, direction)
        if np.linalg.norm(right) < 1e-6:
            # Handle case where direction is parallel to up
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        
        up_corrected = np.cross(direction, right)
        
        # Create rotation matrix
        R = np.column_stack([right, up_corrected, direction])
        
        # Convert rotation matrix to quaternion
        quat = rotation_matrix_to_quaternion(R)
        
        # Create pose [x, y, z, qw, qx, qy, qz]
        pose = np.array([*position, *quat])
        viewpoints.append(pose)
    
    return viewpoints


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion [w, x, y, z].
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion as [w, x, y, z]
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def generate_synthetic_image(pose: np.ndarray, size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Generate a synthetic RGB image based on camera pose.
    
    Args:
        pose: Camera pose [x, y, z, qw, qx, qy, qz]
        size: Image size (height, width)
        
    Returns:
        RGB tensor of shape (3, H, W) with values in [0, 1]
    """
    # Create a simple synthetic scene based on pose
    h, w = size
    
    # Use pose to create position-dependent patterns
    x, y, z = pose[:3]
    
    # Create RGB channels with position-dependent patterns
    r_channel = np.sin(x * 2 * np.pi) * 0.5 + 0.5
    g_channel = np.sin(y * 2 * np.pi) * 0.5 + 0.5
    b_channel = np.sin(z * 2 * np.pi) * 0.5 + 0.5
    
    # Create spatial patterns
    y_coords, x_coords = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
    
    # Combine position and spatial information
    r_img = r_channel * (1 + 0.5 * np.sin(x_coords * 4 * np.pi + x)) * 0.5
    g_img = g_channel * (1 + 0.5 * np.sin(y_coords * 4 * np.pi + y)) * 0.5
    b_img = b_channel * (1 + 0.5 * np.sin((x_coords + y_coords) * 4 * np.pi + z)) * 0.5
    
    # Clip values to [0, 1]
    r_img = np.clip(r_img, 0, 1)
    g_img = np.clip(g_img, 0, 1)
    b_img = np.clip(b_img, 0, 1)
    
    # Stack channels and convert to tensor
    rgb_array = np.stack([r_img, g_img, b_img], axis=0)
    rgb_tensor = torch.from_numpy(rgb_array).float()
    
    return rgb_tensor


class ViewpointScanner:
    """Scanner for systematic viewpoint analysis."""
    
    def __init__(self, model_name: str = "vit_base_patch16_224"):
        """
        Initialize the viewpoint scanner.
        
        Args:
            model_name: Name of the vision model to use
        """
        self.model = get_global_model()
        self.results = []
    
    def scan_viewpoints(
        self,
        num_views: int,
        radius: float = 1.0,
        center: np.ndarray = None,
        elevation_range: Tuple[float, float] = (10.0, 170.0)
    ) -> List[Dict[str, Any]]:
        """
        Scan viewpoints on a sphere and compute entropy for each.
        
        Args:
            num_views: Number of viewpoints to scan
            radius: Radius of the viewing sphere
            center: Center point to look at
            elevation_range: Range of elevation angles
            
        Returns:
            List of results with pose and entropy information
        """
        if center is None:
            center = np.array([0.0, 0.0, 0.0])
        
        # Generate viewpoints
        viewpoints = generate_sphere_viewpoints(
            num_views, radius, center, elevation_range
        )
        
        results = []
        
        print(f"Scanning {num_views} viewpoints...")
        
        for i, pose in enumerate(tqdm(viewpoints, desc="Computing entropy")):
            # Generate synthetic image for this viewpoint
            rgb_tensor = generate_synthetic_image(pose)
            
            # Compute entropy
            entropy, heatmap = rgb_entropy(rgb_tensor, self.model)
            
            # Store results
            result = {
                'index': i,
                'pose_x': pose[0],
                'pose_y': pose[1],
                'pose_z': pose[2],
                'quat_w': pose[3],
                'quat_x': pose[4],
                'quat_y': pose[5],
                'quat_z': pose[6],
                'entropy': entropy,
                'heatmap_mean': np.mean(heatmap),
                'heatmap_std': np.std(heatmap),
                'heatmap_max': np.max(heatmap),
                'heatmap_min': np.min(heatmap)
            }
            results.append(result)
        
        self.results = results
        return results
    
    def save_results(self, output_path: str) -> None:
        """
        Save scan results to CSV file.
        
        Args:
            output_path: Path to save the CSV file
        """
        if not self.results:
            print("No results to save. Run scan_viewpoints first.")
            return
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save to CSV
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        
        print(f"Results saved to: {output_file}")
        print(f"Total viewpoints: {len(self.results)}")
        
        # Print summary statistics
        entropy_values = df['entropy'].values
        print(f"Entropy statistics:")
        print(f"  Mean: {np.mean(entropy_values):.4f}")
        print(f"  Std:  {np.std(entropy_values):.4f}")
        print(f"  Min:  {np.min(entropy_values):.4f}")
        print(f"  Max:  {np.max(entropy_values):.4f}")
    
    def get_best_viewpoints(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get the viewpoints with highest entropy.
        
        Args:
            top_k: Number of top viewpoints to return
            
        Returns:
            List of top viewpoints sorted by entropy
        """
        if not self.results:
            return []
        
        # Sort by entropy (descending)
        sorted_results = sorted(self.results, key=lambda x: x['entropy'], reverse=True)
        
        return sorted_results[:top_k]


def main():
    """Main entry point for viewpoint scanning."""
    parser = argparse.ArgumentParser(description="Scan viewpoints and compute entropy")
    parser.add_argument("--n", type=int, default=180, help="Number of viewpoints to scan")
    parser.add_argument("--out", type=str, default="results/grid.csv", help="Output CSV file")
    parser.add_argument("--radius", type=float, default=1.0, help="Viewing sphere radius")
    parser.add_argument("--center", type=float, nargs=3, default=[0.0, 0.0, 0.0], 
                        help="Center point [x y z]")
    parser.add_argument("--elevation", type=float, nargs=2, default=[10.0, 170.0],
                        help="Elevation range [min max] in degrees")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224",
                        help="Vision model name")
    
    args = parser.parse_args()
    
    # Create scanner
    scanner = ViewpointScanner(args.model)
    
    # Convert center to numpy array
    center = np.array(args.center)
    
    # Scan viewpoints
    results = scanner.scan_viewpoints(
        num_views=args.n,
        radius=args.radius,
        center=center,
        elevation_range=tuple(args.elevation)
    )
    
    # Save results
    scanner.save_results(args.out)
    
    # Show best viewpoints
    best_viewpoints = scanner.get_best_viewpoints(top_k=5)
    print("\nTop 5 viewpoints by entropy:")
    for i, vp in enumerate(best_viewpoints):
        print(f"{i+1}. Entropy: {vp['entropy']:.4f}, "
              f"Pose: ({vp['pose_x']:.2f}, {vp['pose_y']:.2f}, {vp['pose_z']:.2f})")


if __name__ == "__main__":
    main() 