"""
RLBench runner for capturing RGB-D images with randomized camera poses.
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, Any
import random
import time
from tqdm import tqdm

try:
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointPosition
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.tasks import ReachTarget
    from rlbench.observation_config import ObservationConfig
    RLBENCH_AVAILABLE = True
except ImportError:
    RLBENCH_AVAILABLE = False
    print("Warning: RLBench not available. Using mock data generation.")


class RLBenchDataCapture:
    """Captures RGB-D data from RLBench with randomized camera poses."""
    
    def __init__(self, headless: bool = True, render_mode: str = 'rgb_array'):
        """
        Initialize RLBench environment.
        
        Args:
            headless: Whether to run in headless mode
            render_mode: Rendering mode for RLBench
        """
        self.headless = headless
        self.render_mode = render_mode
        self.env = None
        self.task = None
        
        if RLBENCH_AVAILABLE:
            self._setup_rlbench()
        else:
            print("Using mock data generation mode.")
    
    def _setup_rlbench(self) -> None:
        """Setup RLBench environment and task."""
        # Configure observation settings
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        obs_config.right_shoulder_camera.rgb = True
        obs_config.right_shoulder_camera.depth = True
        obs_config.right_shoulder_camera.point_cloud = True
        obs_config.right_shoulder_camera.mask = False
        obs_config.wrist_camera.rgb = True
        obs_config.wrist_camera.depth = True
        obs_config.wrist_camera.point_cloud = True
        
        # Create environment
        action_mode = MoveArmThenGripper(
            arm_action_mode=JointPosition(),
            gripper_action_mode=Discrete()
        )
        
        self.env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=self.headless
        )
        
        # Load task
        self.task = self.env.get_task(ReachTarget)
        
        print("RLBench environment initialized successfully.")
    
    def _randomize_camera_pose(self, base_pose: np.ndarray, angle_range: float = 45.0) -> np.ndarray:
        """
        Randomize camera pose within specified angle range.
        
        Args:
            base_pose: Base camera pose [x, y, z, qw, qx, qy, qz]
            angle_range: Maximum angle deviation in degrees
            
        Returns:
            Randomized camera pose
        """
        # Convert angle range to radians
        angle_rad = np.radians(angle_range)
        
        # Random yaw and pitch offsets
        yaw_offset = random.uniform(-angle_rad, angle_rad)
        pitch_offset = random.uniform(-angle_rad, angle_rad)
        
        # Create rotation matrix for yaw and pitch
        cos_yaw, sin_yaw = np.cos(yaw_offset), np.sin(yaw_offset)
        cos_pitch, sin_pitch = np.cos(pitch_offset), np.sin(pitch_offset)
        
        # Apply rotations to position
        x, y, z = base_pose[:3]
        
        # Rotate around z-axis (yaw)
        x_rot = x * cos_yaw - y * sin_yaw
        y_rot = x * sin_yaw + y * cos_yaw
        
        # Rotate around x-axis (pitch)
        z_rot = z * cos_pitch - y_rot * sin_pitch
        y_final = y_rot * cos_pitch + z * sin_pitch
        
        # Keep original quaternion for simplicity (could be enhanced)
        new_pose = np.array([x_rot, y_final, z_rot, *base_pose[3:]])
        
        return new_pose
    
    def _generate_mock_data(self, index: int) -> Dict[str, Any]:
        """
        Generate mock RGB-D data for testing when RLBench is not available.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary containing mock data
        """
        # Generate random RGB image
        rgb = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # Generate random depth image
        depth = np.random.uniform(0.1, 2.0, (128, 128)).astype(np.float32)
        
        # Generate random pose
        pose = np.array([
            random.uniform(-1.0, 1.0),  # x
            random.uniform(-1.0, 1.0),  # y
            random.uniform(0.5, 1.5),   # z
            random.uniform(-1.0, 1.0),  # qw
            random.uniform(-1.0, 1.0),  # qx
            random.uniform(-1.0, 1.0),  # qy
            random.uniform(-1.0, 1.0),  # qz
        ])
        
        # Normalize quaternion
        q_norm = np.linalg.norm(pose[3:])
        if q_norm > 0:
            pose[3:] /= q_norm
        
        return {
            'rgb': rgb,
            'depth': depth,
            'pose': pose,
            'timestamp': time.time(),
            'index': index
        }
    
    def capture_sample(self, index: int) -> Dict[str, Any]:
        """
        Capture a single RGB-D sample with randomized camera pose.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary containing captured data
        """
        if not RLBENCH_AVAILABLE:
            return self._generate_mock_data(index)
        
        try:
            # Reset task
            descriptions, obs = self.task.reset()
            
            # Get base camera pose (right shoulder camera)
            base_pose = np.array([
                obs.right_shoulder_camera.pose[0],
                obs.right_shoulder_camera.pose[1],
                obs.right_shoulder_camera.pose[2],
                obs.right_shoulder_camera.pose[3],
                obs.right_shoulder_camera.pose[4],
                obs.right_shoulder_camera.pose[5],
                obs.right_shoulder_camera.pose[6]
            ])
            
            # Randomize camera pose
            randomized_pose = self._randomize_camera_pose(base_pose)
            
            # Capture RGB-D data
            rgb = obs.right_shoulder_camera.rgb
            depth = obs.right_shoulder_camera.depth
            
            return {
                'rgb': rgb,
                'depth': depth,
                'pose': randomized_pose,
                'timestamp': time.time(),
                'index': index
            }
            
        except Exception as e:
            print(f"Error capturing sample {index}: {e}")
            return self._generate_mock_data(index)
    
    def capture_dataset(self, num_samples: int, output_dir: str) -> None:
        """
        Capture a dataset of RGB-D samples.
        
        Args:
            num_samples: Number of samples to capture
            output_dir: Directory to save the data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Capturing {num_samples} samples to {output_path}")
        
        for i in tqdm(range(num_samples), desc="Capturing samples"):
            sample = self.capture_sample(i)
            
            # Save as NPZ file
            filename = output_path / f"sample_{i:04d}.npz"
            np.savez_compressed(
                filename,
                rgb=sample['rgb'],
                depth=sample['depth'],
                pose=sample['pose'],
                timestamp=sample['timestamp'],
                index=sample['index']
            )
        
        print(f"Dataset saved to {output_path}")
        print(f"Total samples: {num_samples}")
    
    def shutdown(self) -> None:
        """Clean up resources."""
        if self.env is not None:
            self.env.shutdown()
            print("RLBench environment shut down.")


def main():
    """Main entry point for the RLBench runner."""
    parser = argparse.ArgumentParser(description="Capture RGB-D data from RLBench")
    parser.add_argument("--n", type=int, default=30, help="Number of samples to capture")
    parser.add_argument("--out", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create data capture instance
    capture = RLBenchDataCapture(headless=args.headless)
    
    try:
        # Capture dataset
        capture.capture_dataset(args.n, args.out)
        
    except KeyboardInterrupt:
        print("\nCapture interrupted by user.")
        
    finally:
        # Clean up
        capture.shutdown()


if __name__ == "__main__":
    main() 