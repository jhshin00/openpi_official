#!/usr/bin/env python3
"""
UR3 Policy Test Script
훈련된 UR3 model을 test하기 위한 script
main.py의 작동원리를 참고하여 개선
"""

import dataclasses
import enum
import logging
import pathlib
import time
from typing import Optional

import numpy as np
import rich
import tqdm
import tyro
from openpi_client import websocket_client_policy as _websocket_client_policy

logger = logging.getLogger(__name__)


class TestMode(enum.Enum):
    """Test modes for UR3 policy."""
    RANDOM = "random"      # Random observation으로 test
    REAL_DATA = "real"     # 실제 데이터로 test
    HYBRID = "hybrid"      # 둘 다 test


@dataclasses.dataclass
class Args:
    """Command line arguments."""
    
    # Host and port to connect to the server.
    host: str = "localhost"
    port: int = 8000
    api_key: Optional[str] = None
    
    # Test configuration
    test_mode: TestMode = TestMode.RANDOM
    num_steps: int = 20
    
    # Real data configuration
    data_dir: Optional[pathlib.Path] = None
    task_name: str = "ur3_task"
    
    # Output configuration
    timing_file: Optional[pathlib.Path] = None
    save_actions: bool = False
    actions_file: Optional[pathlib.Path] = None


class TimingRecorder:
    """Records timing measurements for different keys."""
    
    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}
    
    def record(self, key: str, time_ms: float) -> None:
        """Record a timing measurement for the given key."""
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)
    
    def get_stats(self, key: str) -> dict[str, float]:
        """Get statistics for the given key."""
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p25": float(np.quantile(times, 0.25)),
            "p50": float(np.quantile(times, 0.50)),
            "p75": float(np.quantile(times, 0.75)),
            "p90": float(np.quantile(times, 0.90)),
            "p95": float(np.quantile(times, 0.95)),
            "p99": float(np.quantile(times, 0.99)),
        }
    
    def print_all_stats(self) -> None:
        """Print statistics for all keys in a concise format."""
        
        table = rich.table.Table(
            title="[bold blue]UR3 Policy Timing Statistics[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
            title_justify="center",
        )
        
        # Add metric column with custom styling
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)
        
        # Add statistical columns with consistent styling
        stat_columns = [
            ("Mean", "yellow", "mean"),
            ("Std", "yellow", "std"),
            ("P25", "magenta", "p25"),
            ("P50", "magenta", "p50"),
            ("P75", "magenta", "p75"),
            ("P90", "magenta", "p90"),
            ("P95", "magenta", "p95"),
            ("P99", "magenta", "p99"),
        ]
        
        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)
        
        # Add rows for each metric with formatted values
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)
        
        # Print with custom console settings
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)
    
    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        logger.info(f"Writing timings to {path}")
        try:
            import polars as pl
            frame = pl.DataFrame(self._timings)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(path)
            logger.info(f"✅ Timings saved to {path}")
        except ImportError:
            logger.warning("polars not available, skipping parquet export")
        except Exception as e:
            logger.error(f"Failed to save timings: {e}")


def _parse_task_name(task_name: str) -> str:
    """Parse task name from TASK_X_description format to clean description.
    
    Args:
        task_name: Task name like "TASK_1_pick_up_the_bread_and_place_it_on_the_plate"
        
    Returns:
        Clean task description like "pick up the bread and place it on the plate"
    """
    # Remove TASK_X_ prefix
    if task_name.startswith("TASK_"):
        # Find the first underscore after TASK_X_
        parts = task_name.split("_")
        if len(parts) >= 3:
            # Skip TASK and number, join the rest
            description = "_".join(parts[2:])
            # Replace underscores with spaces for better readability
            return description.replace("_", " ")
    
    # If no TASK_X_ prefix, just replace underscores with spaces
    return task_name.replace("_", " ")


class UR3DataLoader:
    """Load real UR3 data for testing from ur3_datasets h5py files."""
    
    def __init__(self, data_dir: pathlib.Path, task_name: str):
        self.data_dir = data_dir
        self.task_name = task_name
        self.data_files = self._find_data_files()
        self.current_file_idx = 0
        
    def _find_data_files(self) -> list[pathlib.Path]:
        """Find all data.hdf5 files in the task directory."""
        data_files = []
        task_dir = self.data_dir / self.task_name
        
        if not task_dir.exists():
            logger.warning(f"Task directory {task_dir} does not exist")
            return []
        
        # Look for data.hdf5 files in subdirectories
        for file_path in task_dir.rglob("data.hdf5"):
            data_files.append(file_path)
        
        logger.info(f"Found {len(data_files)} data files for task: {self.task_name}")
        return sorted(data_files)
    
    def get_next_observation(self) -> dict:
        """Get next observation from real data."""
        if not self.data_files:
            logger.warning("No data files found, falling back to random observation")
            return self._random_observation()
        
        try:
            import h5py
            
            # Get current data file
            data_file = self.data_files[self.current_file_idx % len(self.data_files)]
            
            with h5py.File(data_file, 'r') as f:
                # Load observation data from the 'data' group
                data_group = f['data']
                
                # Get sequence length for random sampling
                seq_length = data_group['joint_positions'].shape[0] if 'joint_positions' in data_group else 1
                
                # Randomly sample a frame from the trajectory
                if seq_length > 1:
                    frame_idx = np.random.randint(0, seq_length)
                else:
                    frame_idx = 0
                
                # Get base and wrist images
                base_image = data_group['base_rgb'][frame_idx, :] if 'base_rgb' in data_group else np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
                wrist_image = data_group['wrist_rgb'][frame_idx, :] if 'wrist_rgb' in data_group else np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
                
                # Convert BGR to RGB (hdf5 files store images in BGR format)
                base_image = base_image[:, :, ::-1]  # Reverse the last dimension (BGR -> RGB)
                wrist_image = wrist_image[:, :, ::-1]  # Reverse the last dimension (BGR -> RGB)
                
                # Get state information (joint positions only)
                joint_positions = data_group['joint_positions'][frame_idx, :] if 'joint_positions' in data_group else np.random.rand(7).astype(np.float32)
                
                # Use only joint positions for state (7 dimensions)
                state = joint_positions
                
                # Get ground truth action for comparison
                # Collect 50 actions from current frame onwards for comparison with policy output (50, 7)
                ground_truth_actions = []
                remaining_frames = seq_length - frame_idx
                
                if remaining_frames >= 50:
                    # Enough frames to get 50 actions
                    for i in range(50):
                        action = data_group['actions'][frame_idx + i, :]
                        ground_truth_actions.append(action)
                else:
                    # Not enough frames, get available actions and repeat the last one
                    for i in range(remaining_frames):
                        action = data_group['actions'][frame_idx + i, :]
                        ground_truth_actions.append(action)
                    
                    # Repeat the last action to fill up to 50
                    last_action = ground_truth_actions[-1] if ground_truth_actions else np.zeros(7)
                    while len(ground_truth_actions) < 50:
                        ground_truth_actions.append(last_action)
                
                # Convert to numpy array with shape (50, 7)
                ground_truth_actions = np.array(ground_truth_actions)
                
                # Parse task name to get clean prompt
                task_prompt = _parse_task_name(self.task_name)
                
                # Move to next file for variety
                self.current_file_idx += 1
                
                return {
                    "observation/base_image": base_image,
                    "observation/wrist_image": wrist_image,
                    "observation/state": state,
                    "prompt": task_prompt,
                    "ground_truth_action": ground_truth_actions,
                    "frame_idx": frame_idx,
                    "seq_length": seq_length,
                    "remaining_frames": remaining_frames
                }
                
        except Exception as e:
            logger.warning(f"Failed to load real data: {e}, using random observation")
            return self._random_observation()
    
    def _random_observation(self) -> dict:
        """Fallback to random observation."""
        # Create random ground truth actions with shape (50, 7)
        random_actions = np.random.rand(50, 7).astype(np.float32)
        
        return {
            "observation/base_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/state": np.random.rand(7).astype(np.float32),  # 7 joint positions only
            "prompt": _parse_task_name(self.task_name) if hasattr(self, 'task_name') else "pick up the object and place it on the table",
            "ground_truth_action": random_actions,
            "frame_idx": 0,
            "seq_length": 1,
            "remaining_frames": 1
        }


def test_ur3_policy(args: Args) -> None:
    """Test UR3 policy with various observation sources."""
    
    print("🤖 UR3 Policy Test Script")
    print("=" * 60)
    print(f"🔌 Connecting to policy server at {args.host}:{args.port}")
    print(f"🧪 Test mode: {args.test_mode.value}")
    print(f"📊 Number of steps: {args.num_steps}")
    
    # Initialize data loader if using real data
    data_loader = None
    if args.test_mode in [TestMode.REAL_DATA, TestMode.HYBRID]:
        if args.data_dir is None:
            logger.warning("Real data mode selected but no data_dir provided, using random data")
            args.test_mode = TestMode.RANDOM
        else:
            data_loader = UR3DataLoader(args.data_dir, args.task_name)
    
    try:
        # Connect to policy server
        policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
        )
        
        # Get server metadata
        metadata = policy.get_server_metadata()
        print(f"✅ Connected to policy server!")
        print(f"📋 Server metadata: {metadata}")
        
        # Warm up the model
        print("🔥 Warming up the model...")
        for _ in range(2):
            if data_loader:
                obs = data_loader.get_next_observation()
            else:
                obs = _random_observation_ur3()
            policy.infer(obs)
        
        # Initialize timing recorder
        timing_recorder = TimingRecorder()
        all_actions = []
        
        # Run the test
        print(f"🚀 Running {args.num_steps} test steps...")
        
        for step in tqdm.trange(args.num_steps, desc="Testing UR3 Policy"):
            inference_start = time.time()
            
            # Get observation based on test mode
            if args.test_mode == TestMode.RANDOM:
                obs = _random_observation_ur3()
                obs_source = "random"
            elif args.test_mode == TestMode.REAL_DATA:
                obs = data_loader.get_next_observation()
                obs_source = "real_data"
            else:  # HYBRID
                if step % 2 == 0:
                    obs = _random_observation_ur3()
                    obs_source = "random"
                else:
                    obs = data_loader.get_next_observation()
                    obs_source = "real_data"
            
            # Run policy inference
            result = policy.infer(obs)
            
            # Record timing
            inference_time = 1000 * (time.time() - inference_start)
            timing_recorder.record("client_infer_ms", inference_time)
            
            # Record server timing if available
            for key, value in result.get("server_timing", {}).items():
                timing_recorder.record(f"server_{key}", value)
            for key, value in result.get("policy_timing", {}).items():
                timing_recorder.record(f"policy_{key}", value)
            
            # Store actions if requested
            if args.save_actions:
                actions = result["actions"]
                all_actions.append({
                    "step": step,
                    "source": obs_source,
                    "actions": actions,
                    "prompt": obs.get("prompt", "unknown"),
                    "inference_time_ms": inference_time
                })
            
            # Print step info
            actions = result["actions"]
            ground_truth = obs.get("ground_truth_action")
            frame_info = f"Frame {obs.get('frame_idx', 0)}/{obs.get('seq_length', 1)}"
            remaining_info = f"Remaining: {obs.get('remaining_frames', 0)}"
            
            print(f"Step {step+1:3d}: {obs_source:>10} | {frame_info:>12} | {remaining_info:>10} | "
                  f"Actions: {actions.shape} | "
                  f"Time: {inference_time:.1f}ms | "
                  f"Prompt: {obs.get('prompt', 'unknown')[:30]}...")
            
            # Compare policy actions with ground truth if available
            if ground_truth is not None:
                print(f"  📊 Policy Action: {actions.flatten()[:5].tolist()}... (shape: {actions.shape})")
                print(f"  🎯 Ground Truth: {ground_truth.flatten()[:5].tolist()}... (shape: {ground_truth.shape})")
                
                # Calculate action difference if shapes match
                if actions.shape == ground_truth.shape:
                    action_diff = np.abs(actions - ground_truth)
                    mean_diff = np.mean(action_diff)
                    max_diff = np.max(action_diff)
                    print(f"  📈 Action Difference - Mean: {mean_diff:.4f}, Max: {max_diff:.4f}")
                    
                    # Show first few action comparisons
                    print(f"  🔍 First 3 actions comparison:")
                    for i in range(min(3, actions.shape[0])):
                        policy_action = actions[i, :5].tolist()  # First 5 values
                        gt_action = ground_truth[i, :5].tolist()  # First 5 values
                        diff = np.mean(np.abs(actions[i] - ground_truth[i]))
                        print(f"    Action {i}: Policy {policy_action}... vs GT {gt_action}... (diff: {diff:.4f})")
                else:
                    print(f"  ⚠️  Shape mismatch: Policy {actions.shape} vs GT {ground_truth.shape}")
                print()
            else:
                print(f"  📊 Policy Action: {actions.flatten()[:5].tolist()}... (shape: {actions.shape})")
                print()
        
        print("✅ Test completed successfully!")
        
        # Print timing statistics
        print("\n" + "="*60)
        timing_recorder.print_all_stats()
        
        # Save timings if requested
        if args.timing_file:
            timing_recorder.write_parquet(args.timing_file)
        
        # Save actions if requested
        if args.save_actions and args.actions_file:
            _save_actions(all_actions, args.actions_file)
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


def _random_observation_ur3() -> dict:
    """Create random UR3 observation (same as main.py)."""
    return {
        "observation/state": np.random.rand(7),
        "observation/base_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _save_actions(actions: list, file_path: pathlib.Path) -> None:
    """Save actions to file."""
    try:
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_actions = []
        for action_data in actions:
            serializable_action = {
                "step": action_data["step"],
                "source": action_data["source"],
                "prompt": action_data["prompt"],
                "inference_time_ms": action_data["inference_time_ms"],
                "actions_shape": list(action_data["actions"].shape),
                "actions_sample": action_data["actions"].flatten()[:10].tolist()  # First 10 values
            }
            serializable_actions.append(serializable_action)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(serializable_actions, f, indent=2)
        
        logger.info(f"✅ Actions saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to save actions: {e}")


def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO)
    
    args = tyro.cli(Args)
    
    # Validate arguments
    if args.test_mode in [TestMode.REAL_DATA, TestMode.HYBRID] and args.data_dir is None:
        print("❌ Error: data_dir must be provided for real_data or hybrid mode")
        return
    
    if args.data_dir and not args.data_dir.exists():
        print(f"❌ Error: data_dir {args.data_dir} does not exist")
        return
    
    # Run the test
    test_ur3_policy(args)


if __name__ == "__main__":
    main()