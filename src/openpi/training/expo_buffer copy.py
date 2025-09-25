"""
EXPO Replay Buffer with Libero Offline Data Support
==================================================

This module provides a trajectory-based replay buffer that can optionally preload
offline demonstration data from the LIBERO-90 dataset for robot manipulation tasks.

Key Features:
- Trajectory-based storage and sampling
- Automatic loading of LIBERO-90 offline data
- Support for multi-modal observations (RGB images, robot states)
- Efficient trajectory removal when buffer capacity is exceeded

LIBERO-90 Dataset:
- 90 different manipulation tasks across 3 scenes (Kitchen, Living Room, Study)
- ~4500 demonstration trajectories total
- Multi-modal observations: RGB images + robot proprioception
- Sparse reward structure (1 at goal completion, 0 otherwise)

Data Format:
- HDF5 files with nested structure: data/demo_N/{actions, rewards, dones, obs/...}
- Observations include: agentview_rgb (256x256), eye_in_hand_rgb (256x256), ee_pos, ee_ori, joint_states, etc.
- Actions: 7D vector [x, y, z, qx, qy, qz, qw, gripper]
- Rewards: Sparse binary rewards (1 at task completion)
- Masks: Continuation flags (1 - dones)

Usage:
    buffer = TrajReplayBuffer(obs_space, action_space, capacity, use_offline_data=True)
    # Automatically loads all LIBERO-90 trajectories
    batch = buffer.sample(batch_size=256)  # Sample random steps
    trajs = buffer.get_random_trajs(num_trajs=32)  # Sample random trajectories
"""

from typing import Union
from typing import Iterable, Optional
import jax 
import jax.numpy as jnp
import gym
import gym.spaces
import numpy as np
import pickle

import copy
import os
import h5py

import collections
from openpi.data.dataset import Dataset, DatasetDict

# Import required modules for expo_pi0 format conversion
from openpi.transforms import TokenizePrompt
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi_client import image_tools

# Import helper functions from train_utils
from openpi.training.expo_train_utils import pad_to_dim

def _init_replay_dict(
    obs_space: gym.Space,
    capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    buffer_data: Union[np.ndarray, dict], 
    input_data: Union[np.ndarray, dict], 
    insert_index: int
):
    """Recursively insert data into buffer structure."""
    if isinstance(buffer_data, np.ndarray) and isinstance(input_data, np.ndarray):
        # Both are numpy arrays - direct assignment
        buffer_data[insert_index] = input_data
    elif (hasattr(buffer_data, 'items') and hasattr(buffer_data, 'keys') and 
          hasattr(input_data, 'items') and hasattr(input_data, 'keys')):
        # Both are dictionary-like objects - recurse into subkeys
        for key in input_data:
            if key in buffer_data:
                _insert_recursively(buffer_data[key], input_data[key], insert_index)
    else:
        # Type mismatch - direct assignment (for non-array types)
        if isinstance(buffer_data, np.ndarray):
            buffer_data[insert_index] = input_data
        else:
            raise TypeError(f"Cannot insert {type(input_data)} into {type(buffer_data)}")


def _sample_recursively(
    buffer_data: Union[np.ndarray, dict], 
    indices: np.ndarray
) -> Union[np.ndarray, dict]:
    """Recursively sample data from buffer structure."""
    if isinstance(buffer_data, np.ndarray):
        # Numpy array - sample by indices
        return buffer_data[indices]
    elif hasattr(buffer_data, 'items') and hasattr(buffer_data, 'keys'):
        # Dictionary-like object - recurse into subkeys
        result = {}
        for key, value in buffer_data.items():
            result[key] = _sample_recursively(value, indices)
        return result
    else:
        # Non-array type - return as is
        return buffer_data


def libero_obs_to_expo_pi0_format(libero_obs, task_description, max_token_len=48, action_dim=32):
    """
    Convert libero observation format to expo_pi0 format.
    
    Args:
        libero_obs: Dictionary with libero observation keys
        task_description: Task description string
        max_token_len: Maximum token length for prompt
    
    Returns:
        Dictionary in expo_pi0 format
    """
    # Convert images from libero format to expo_pi0 format
    base_img = np.ascontiguousarray(libero_obs["agentview_rgb"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(libero_obs["eye_in_hand_rgb"][::-1, ::-1])
    
    # Resize to 224x224 (expo_pi0 format)
    base_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(base_img, 224, 224)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, 224, 224)
    )
    
    images = {
        "base_0_rgb": base_img[None, ...],
        "left_wrist_0_rgb": wrist_img[None, ...],
        "right_wrist_0_rgb": np.zeros_like(base_img)[None, ...],
    }
    image_masks = {
        "base_0_rgb": np.array([True]),
        "left_wrist_0_rgb": np.array([True]),
        "right_wrist_0_rgb": np.array([False]),
    }

    # Convert state from libero format to expo_pi0 format
    # libero: ee_pos (3) + ee_ori (3) + gripper_states (2)
    # expo_pi0: ee_pos (3) + ee_ori (3) + gripper_states (1) - remove last gripper dim
    state = np.concatenate([
        libero_obs["ee_pos"],  # (3,)
        libero_obs["ee_ori"],  # (3,) - already in axis-angle format
        libero_obs["gripper_states"][:1]  # (1,) - take only first gripper joint
    ])[None, ...].astype(np.float32)

    state = pad_to_dim(state, action_dim, axis=-1)
    
    # Tokenize prompt
    prompt = str(task_description)
    tokenizer = _tokenizer.PaligemmaTokenizer(max_token_len)
    obs_dict = TokenizePrompt(tokenizer)({
        "image": images,
        "image_mask": image_masks,
        "state": state,
        "prompt": prompt,
    })
    obs_dict["tokenized_prompt"] = obs_dict["tokenized_prompt"][None, ...]
    obs_dict["tokenized_prompt_mask"] = obs_dict["tokenized_prompt_mask"][None, ...]
    
    return obs_dict


def load_libero_trajectory(hdf5_path: str, demo_idx: int = 0, task_description: str = "manipulation task", max_token_len: int = 48, action_dim: int = 32) -> DatasetDict:
    """
    Load a single trajectory from libero HDF5 file and convert to expected format.
    
    Libero HDF5 Data Structure:
    ==========================
    HDF5 File Structure:
    - data/
      - demo_0/
        - actions: (T, 7) - Robot actions [x, y, z, qx, qy, qz, qw, gripper]
        - rewards: (T,) - Reward values (usually sparse, 1 at goal, 0 otherwise)
        - dones: (T,) - Episode termination flags (1 at episode end, 0 otherwise)
        - obs/
          - agentview_rgb: (T, 256, 256, 3) - Third-person camera RGB images
          - eye_in_hand_rgb: (T, 256, 256, 3) - Wrist-mounted camera RGB images
          - ee_pos: (T, 3) - End-effector position [x, y, z]
          - ee_ori: (T, 3) - End-effector orientation [rx, ry, rz] (euler angles)
          - ee_states: (T, 6) - Combined end-effector state [pos + ori]
          - gripper_states: (T, 2) - Gripper joint positions
          - joint_states: (T, 7) - Robot joint positions
        - robot_states: (T, ...) - Additional robot state information
        - states: (T, ...) - Environment state information
      - demo_1/
        - ... (same structure as demo_0)
      - ...
      - demo_N/
        - ... (same structure as demo_0)
    
    Converted Trajectory Format (for insert_traj):
    =============================================
    {
        'episode_length': int,  # Number of timesteps in trajectory
        'observations': {
            'image': {
                'base_0_rgb': list,         # List of (1, 224, 224, 3) arrays
                'left_wrist_0_rgb': list,   # List of (1, 224, 224, 3) arrays  
                'right_wrist_0_rgb': list,  # List of (1, 224, 224, 3) arrays
            },
            'image_mask': {
                'base_0_rgb': list,         # List of (1,) boolean arrays
                'left_wrist_0_rgb': list,   # List of (1,) boolean arrays
                'right_wrist_0_rgb': list,  # List of (1,) boolean arrays
            },
            'state': list,                  # List of (1, 7) arrays [ee_pos(3) + ee_ori(3) + gripper(2)]
            'tokenized_prompt': list,       # List of (1, max_token_len) arrays
            'tokenized_prompt_mask': list,  # List of (1, max_token_len) boolean arrays
        },
        'actions': np.ndarray,     # (T, 7) - Robot actions
        'rewards': np.ndarray,     # (T,) - Reward values
        'masks': np.ndarray,       # (T,) - Continuation masks (1 - dones)
    }
    
    Args:
        hdf5_path: Path to the HDF5 file
        demo_idx: Index of the demonstration to load (default: 0)
    
    Returns:
        DatasetDict in the format expected by insert_traj
    """
    with h5py.File(hdf5_path, 'r') as f:
        demo = f['data'][f'demo_{demo_idx}']
        
        # Extract data
        actions = demo['actions'][:]  # (T, 7)
        rewards = demo['rewards'][:]  # (T,)
        dones = demo['dones'][:]  # (T,)
        obs_group = demo['obs']
        
        # Extract observations in libero format
        libero_observations = {}
        for key in obs_group.keys():
            libero_observations[key] = obs_group[key][:]  # (T, ...)
        
        # Convert to expo_pi0 format - fully vectorized processing for maximum efficiency
        base_imgs = libero_observations["agentview_rgb"]  # (T, 256, 256, 3)
        wrist_imgs = libero_observations["eye_in_hand_rgb"]  # (T, 256, 256, 3)
        
        # Vectorized image processing - flip and resize all at once
        # Flip images: [::-1, ::-1] for both height and width
        base_imgs_flipped = base_imgs[:, ::-1, ::-1, :]  # (T, 256, 256, 3)
        wrist_imgs_flipped = wrist_imgs[:, ::-1, ::-1, :]  # (T, 256, 256, 3)
        
        # Process all images at once using existing image_tools (supports batch processing!)
        # image_tools.resize_with_pad already handles batch dimension via reshape(-1, ...)
        processed_base_imgs = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(base_imgs_flipped, 224, 224)
        )  # (T, 224, 224, 3)
        
        processed_wrist_imgs = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_imgs_flipped, 224, 224)
        )  # (T, 224, 224, 3)
        
        # Create images dictionary
        images = {
            "base_0_rgb": processed_base_imgs,
            "left_wrist_0_rgb": processed_wrist_imgs,
            "right_wrist_0_rgb": np.zeros_like(processed_base_imgs),
        }
        image_masks = {
            "base_0_rgb": np.ones(len(actions), dtype=bool),
            "left_wrist_0_rgb": np.ones(len(actions), dtype=bool),
            "right_wrist_0_rgb": np.zeros(len(actions), dtype=bool),
        }
        
        # Convert state from libero format to expo_pi0 format
        # libero: ee_pos (3) + ee_ori (3) + gripper_states (2)
        # expo_pi0: ee_pos (3) + ee_ori (3) + gripper_states (1) - remove last gripper dim
        state = np.concatenate([
            libero_observations["ee_pos"],  # (T, 3)
            libero_observations["ee_ori"],  # (T, 3) - already in axis-angle format
            libero_observations["gripper_states"][:, :1]  # (T, 1) - take only first gripper joint
        ], axis=1).astype(np.float32)  # (T, 7)

        state = pad_to_dim(state, action_dim, axis=-1)
        
        # Tokenize prompt once for the entire trajectory
        prompt = str(task_description)
        tokenizer = _tokenizer.PaligemmaTokenizer(max_token_len)
        
        # Tokenize once - TokenizePrompt only needs prompt
        tokenized = TokenizePrompt(tokenizer)({"prompt": prompt})
        tokenized_prompt = tokenized["tokenized_prompt"]  # (1, max_token_len)
        tokenized_prompt_mask = tokenized["tokenized_prompt_mask"]  # (1, max_token_len)
        
        # Repeat for all timesteps
        tokenized_prompts = np.tile(tokenized_prompt, (len(actions), 1))  # (T, max_token_len)
        tokenized_prompt_masks = np.tile(tokenized_prompt_mask, (len(actions), 1))  # (T, max_token_len)
        
        # Create final observation structure
        expo_observations = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
            "tokenized_prompt": tokenized_prompts,
            "tokenized_prompt_mask": tokenized_prompt_masks,
        }
        
        # Convert dones to masks (masks = 1 - dones)
        masks = (1 - dones).astype(np.bool_)
        
        # Create flat trajectory structure directly (no need for helper function)
        is_success = (rewards[-1] == 1.0)  # Assume success if last reward is 1
        episode_length = len(actions)
        env_steps = len(actions)
        
        # Print trajectory info
        print(f"  Loaded trajectory: {task_description}")
        print(f"    - Episode length: {episode_length}")
        print(f"    - Success: {is_success}")
        print(f"    - Episode return: {rewards.sum()}")
        print(f"    - Action range: [{actions.min():.3f}, {actions.max():.3f}]")
        
        # Create flat trajectory structure directly
        traj = {
            'base_img': expo_observations['image']['base_0_rgb'],  # (T, 224, 224, 3)
            'wrist_img': expo_observations['image']['left_wrist_0_rgb'],  # (T, 224, 224, 3)
            'base_img_mask': expo_observations['image_mask']['base_0_rgb'],  # (T,)
            'wrist_img_mask': expo_observations['image_mask']['left_wrist_0_rgb'],  # (T,)
            'state': expo_observations['state'],  # (T, 7)
            'tokenized_prompt': expo_observations['tokenized_prompt'],  # (T, max_token_len)
            'tokenized_prompt_mask': expo_observations['tokenized_prompt_mask'],  # (T, max_token_len)
            'actions': actions,  # (T, 7)
            'rewards': rewards,  # (T,)
            'masks': masks,  # (T,)
            'is_success': is_success,
            'episode_return': rewards.sum(),
            'episode_length': episode_length,
            'env_steps': env_steps
        }
        
        return traj


def load_sample_libero_trajectories(dataset_dir: str, max_trajectories: int = None, max_token_len: int = 48, action_dim: int = 32) -> list:
    """
    Load a sample of trajectories from libero_90 dataset.
    
    Args:
        dataset_dir: Path to the libero_90 dataset directory
        max_trajectories: Maximum number of trajectories to load (None for all)
        max_token_len: Maximum token length for prompt tokenization
    
    Returns:
        List of trajectory dictionaries, each in the format expected by insert_traj
    """
    trajectories = []
    
    # Get all HDF5 files in the directory
    hdf5_files = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
    hdf5_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(hdf5_files)} HDF5 files. Loading trajectories...")
    
    for file_idx, hdf5_file in enumerate(hdf5_files):
        if max_trajectories is not None and len(trajectories) >= max_trajectories:
            break
            
        hdf5_path = os.path.join(dataset_dir, hdf5_file)
        
        # Extract task description from filename
        # Format: PREFIX_TASK_NAME_demo.hdf5
        task_name = hdf5_file.replace('_demo.hdf5', '')
        
        # Remove uppercase prefixes (e.g., KITCHEN_, SCENE1_, etc.)
        import re
        task_description = re.sub(r'^[A-Z0-9_]+_', '', task_name)
        
        task_description = task_description.replace('_', ' ').lower()
        
        print(f"\nProcessing file {file_idx + 1}/{len(hdf5_files)}: {hdf5_file}")
        print(f"  Task: {task_description}")
        
        # Load demonstrations from this file
        with h5py.File(hdf5_path, 'r') as f:
            demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
            demo_indices = [int(k.split('_')[1]) for k in demo_keys]
            demo_indices.sort()
            
            print(f"  Found {len(demo_indices)} demos in this file")
            
            # Load demos from each file if we have a limit
            if max_trajectories is not None:
                # Calculate how many demos to load from this file
                remaining_trajectories = max_trajectories - len(trajectories)
                remaining_files = len(hdf5_files) - hdf5_files.index(hdf5_file)
                max_demos_per_file = max(1, remaining_trajectories // remaining_files) if remaining_files > 0 else 0
                max_demos_per_file = min(max_demos_per_file, len(demo_indices))
            else:
                max_demos_per_file = len(demo_indices)
            
            print(f"  Loading {max_demos_per_file} demos from this file")
            
            for demo_idx in demo_indices[:max_demos_per_file]:
                if max_trajectories is not None and len(trajectories) >= max_trajectories:
                    break
                    
                try:
                    traj = load_libero_trajectory(hdf5_path, demo_idx, task_description, max_token_len, action_dim)
                    trajectories.append(traj)
                    print(f"  ✓ Total loaded: {len(trajectories)}/{max_trajectories if max_trajectories else 'all'}")
                except Exception as e:
                    print(f"  ✗ Warning: Failed to load {hdf5_file} demo_{demo_idx}: {e}")
                    continue
    
    print(f"Loaded {len(trajectories)} trajectories from {len(hdf5_files)} files")
    return trajectories


def load_all_libero_trajectories(dataset_dir: str, max_token_len: int = 48, action_dim: int = 32) -> list:
    """
    Load all trajectories from libero_90 dataset.
    
    Libero_90 Dataset Structure:
    ===========================
    The libero_90 dataset contains 90 different manipulation tasks across 3 scenes:
    - KITCHEN_SCENE1-10: Kitchen manipulation tasks (50 tasks)
    - LIVING_ROOM_SCENE1-6: Living room manipulation tasks (30 tasks)  
    - STUDY_SCENE1-4: Study room manipulation tasks (10 tasks)
    
    Each task has multiple demonstration files (e.g., *_demo.hdf5), and each file
    contains multiple demonstrations (demo_0, demo_1, ..., demo_N) where N varies
    by task (typically 50 demonstrations per task).
    
    Total dataset size: ~4500 trajectories across 90 tasks
    
    File naming convention:
    - KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5
    - LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket_demo.hdf5
    - STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy_demo.hdf5
    
    Each HDF5 file contains:
    - data/demo_0/ through data/demo_N/ (multiple demonstrations)
    - Each demo follows the structure described in load_libero_trajectory()
    
    Args:
        dataset_dir: Path to the libero_90 dataset directory
        max_token_len: Maximum token length for prompt tokenization
    
    Returns:
        List of trajectory dictionaries, each in the format expected by insert_traj
    """
    trajectories = []
    
    # Get all HDF5 files in the directory
    hdf5_files = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
    hdf5_files.sort()  # Sort for consistent ordering
    
    for hdf5_file in hdf5_files:
        hdf5_path = os.path.join(dataset_dir, hdf5_file)
        
        # Extract task description from filename
        # Format: PREFIX_TASK_NAME_demo.hdf5
        task_name = hdf5_file.replace('_demo.hdf5', '')
        
        # Remove uppercase prefixes (e.g., KITCHEN_, SCENE1_, etc.)
        import re
        task_description = re.sub(r'^[A-Z0-9_]+_', '', task_name)
        
        task_description = task_description.replace('_', ' ').lower()
        
        # Load all demonstrations from this file
        with h5py.File(hdf5_path, 'r') as f:
            demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
            demo_indices = [int(k.split('_')[1]) for k in demo_keys]
            demo_indices.sort()
            
            for demo_idx in demo_indices:
                try:
                    traj = load_libero_trajectory(hdf5_path, demo_idx, task_description, max_token_len, action_dim)
                    trajectories.append(traj)
                except Exception as e:
                    print(f"Warning: Failed to load {hdf5_file} demo_{demo_idx}: {e}")
                    continue
    
    print(f"Loaded {len(trajectories)} trajectories from {len(hdf5_files)} files")
    return trajectories

class TrajReplayBuffer(Dataset):
    """
    Trajectory-based replay buffer that stores complete trajectories as units.
    
    Usage with Libero Data:
    ======================
    # Initialize buffer with offline libero data
    buffer = TrajReplayBuffer(
        observation_space=obs_space,
        action_space=action_space, 
        capacity=1000,  # Number of trajectories, not timesteps
        use_offline_data=True,  # Load libero_90 dataset on initialization
        libero_data_dir="/ssd2/EXPO/datasets/libero_90"
    )
    
    # The buffer will automatically load ~4500 trajectories from libero_90
    # Each trajectory contains:
    # - RGB images (agentview_rgb, eye_in_hand_rgb): (T, 256, 256, 3)
    # - Robot states (ee_pos, ee_ori, joint_states, gripper_states)
    # - Actions: (T, 7) - [x, y, z, qx, qy, qz, qw, gripper]
    # - Rewards: (T,) - Sparse rewards (1 at goal, 0 otherwise)
    # - Masks: (T,) - Continuation masks (1 - dones)
    
    # Sample random trajectories for training
    batch = buffer.get_random_trajs(num_trajs=32)
    
    # Sample random steps for training
    batch = buffer.sample(batch_size=256)
    
    # Add new online trajectories
    buffer.insert_traj(new_trajectory)
    """
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, capacity: int, use_offline_data: bool, libero_data_dir: str = "/ssd2/EXPO/datasets/libero_goal", offline_dataset_subset_num: int = None):
        self.observation_space = observation_space
        self.action_space = action_space
        self.capacity = capacity  # Total number of steps (timesteps), not trajectories
        self.use_offline_data = use_offline_data
        self.libero_data_dir = libero_data_dir
        self.offline_dataset_subset_num = offline_dataset_subset_num  # Number of trajectories to sample from offline dataset

        print("making trajectory replay buffer of capacity ", self.capacity, "steps")

        # Store trajectories as complete units
        self.trajectories = {}  # traj_id -> trajectory data
        self.traj_metadata = {}  # traj_id -> metadata (length, etc.)
        self.size = 0  # Number of trajectories stored
        self.total_steps = 0  # Total number of steps across all trajectories
        self._traj_counter = 0
        self.streaming_buffer_size = None # this is for streaming the online data
        
        # Load offline data if requested
        if self.use_offline_data:
            print("Loading offline libero data...")
            self._load_offline_data()

    def _load_offline_data(self):
        """Load offline libero data into the buffer."""
        try:
            # Load trajectories from libero dataset (with subset if specified)
            if self.offline_dataset_subset_num is not None:
                print(f"\n=== Loading {self.offline_dataset_subset_num} trajectories from libero dataset ===")
                trajectories = load_sample_libero_trajectories(
                    self.libero_data_dir, 
                    max_trajectories=self.offline_dataset_subset_num,
                    max_token_len=48,
                    action_dim=32
                )
            else:
                print("\n=== Loading all trajectories from libero dataset ===")
                trajectories = load_all_libero_trajectories(self.libero_data_dir, 48, 32)
            
            print(f"\n=== Inserting {len(trajectories)} trajectories into buffer ===")
            
            # Insert trajectories into buffer
            loaded_count = 0
            for traj_idx, traj in enumerate(trajectories):
                try:
                    # Check if we have space for this trajectory
                    if self.total_steps + traj['episode_length'] > self.capacity:
                        print(f"Buffer capacity reached. Loaded {loaded_count} trajectories ({self.total_steps} steps).")
                        break
                    
                    self.insert_traj(traj)
                    loaded_count += 1
                    
                    # Print progress every 10 trajectories
                    if loaded_count % 10 == 0:
                        print(f"  Inserted {loaded_count}/{len(trajectories)} trajectories ({self.total_steps} total steps)")
                    
                except Exception as e:
                    print(f"  ✗ Warning: Failed to insert trajectory {traj_idx}: {e}")
                    continue
            
            print(f"\n=== Successfully loaded {loaded_count} trajectories into buffer ===")
            print(f"Buffer size: {self.size} trajectories, {self.total_steps} total steps")
            print(f"Buffer utilization: {self.total_steps / self.capacity * 100:.1f}%")
            
        except Exception as e:
            print(f"Error loading offline data: {e}")
            print("Continuing with empty buffer...")

    def __len__(self) -> int:
        return self.size
    
    def length(self) -> int:
        return self.size

    def insert_traj(self, traj: DatasetDict):
        """
        Insert a complete trajectory into the replay buffer.
        traj should have the structure from collect_trajectory function.
        """
        traj_length = traj['episode_length']
        
        # Check if we need to remove old trajectories to make space
        while self.total_steps + traj_length > self.capacity and self.size > 0:
            # Remove the oldest trajectory (FIFO)
            oldest_traj_id = min(self.trajectories.keys())
            removed_traj = self.trajectories[oldest_traj_id]
            removed_length = self.traj_metadata[oldest_traj_id]['episode_length']
            
            del self.trajectories[oldest_traj_id]
            del self.traj_metadata[oldest_traj_id]
            self.size -= 1
            self.total_steps -= removed_length
        
        # Store the complete trajectory
        traj_id = self._traj_counter
        self.trajectories[traj_id] = traj
        self.traj_metadata[traj_id] = {
            'episode_length': traj_length,
            'is_success': traj.get('is_success', False),
            'episode_return': traj.get('episode_return', 0.0),
            'env_steps': traj.get('env_steps', traj_length)
        }
        
        self.size += 1
        self.total_steps += traj_length
        self._traj_counter += 1




    def get_random_trajs(self, num_trajs: int):
        """Sample random trajectories from the buffer."""
        if self.size == 0:
            return None
            
        available_traj_ids = list(self.trajectories.keys())
        num_trajs = min(num_trajs, len(available_traj_ids))
        selected_traj_ids = np.random.choice(available_traj_ids, num_trajs, replace=False)
        
        trajectories_list = []
        
        for traj_id in selected_traj_ids:
            traj = self.trajectories[traj_id]
            
            # Create next_observations by shifting by 1 timestep
            next_traj = {}
            for k, v in traj.items():
                if k in ['actions', 'rewards', 'masks', 'is_success', 'episode_return', 'episode_length', 'env_steps']:
                    # Copy metadata as is
                    next_traj[k] = v
                else:
                    # Shift observation data by 1 timestep
                    if len(v) > 1:
                        next_traj[k] = np.concatenate([v[1:], v[-1:]], axis=0)
                    else:
                        next_traj[k] = v
            
            # Create trajectory with next observations
            traj_with_next = {
                'observations': traj,
                'next_observations': next_traj,
                'actions': traj['actions'],
                'rewards': traj['rewards'],
                'terminals': 1 - traj['masks'],  # terminals = 1 - masks
                'masks': traj['masks'],
            }
            
            trajectories_list.append(traj_with_next)
        
        return trajectories_list

    def sample(self, batch_size: int, keys: Optional[Iterable[str]] = None, indx: Optional[np.ndarray] = None, action_horizon: int = 50, action_dim: int = 32, discount_factor: float = 0.99):
        """Sample random steps from the buffer with action horizon."""
        if self.size == 0:
            return None
            
        # Sample random trajectories first
        available_traj_ids = list(self.trajectories.keys())
        # Sample batch_size trajectories with replacement to get the desired batch size
        selected_traj_ids = np.random.choice(available_traj_ids, batch_size, replace=True)
        
        observations_list = []
        next_observations_list = []
        actions_list = []
        rewards_list = []
        terminals_list = []
        masks_list = []
        
        for traj_id in selected_traj_ids:
            traj = self.trajectories[traj_id]
            
            # Sample a random timestep from this trajectory
            traj_length = traj['episode_length']
            if traj_length > 1:
                t = np.random.randint(0, traj_length)
            else:
                t = 0
            
            # Extract single timestep observation
            obs = {}
            for k, v in traj.items():
                if k in ['actions', 'rewards', 'masks', 'is_success', 'episode_return', 'episode_length', 'env_steps']:
                    # Skip metadata fields
                    continue
                else:
                    # Extract single timestep
                    obs[k] = v[t]
            
            # Extract action horizon from this timestep
            # If not enough actions remaining, repeat the last action
            remaining_actions = traj['actions'][t:]
            if len(remaining_actions) >= action_horizon:
                action_sequence = remaining_actions[:action_horizon]
            else:
                # Pad with last action if not enough remaining
                last_action = remaining_actions[-1:] if len(remaining_actions) > 0 else traj['actions'][-1:]
                padding_needed = action_horizon - len(remaining_actions)
                action_sequence = np.concatenate([
                    remaining_actions,
                    np.tile(last_action, (padding_needed, 1))
                ])
            
            # Pad action from 7D to action_dim (32D) - first 7 dimensions are original action, rest are zeros
            if action_dim > 7:
                padding = np.zeros((action_horizon, action_dim - 7))
                action_sequence_padded = np.concatenate([action_sequence, padding], axis=1)
            else:
                action_sequence_padded = action_sequence
            
            # Extract next observation after action_horizon steps
            next_t = min(t + action_horizon, traj_length - 1)
            next_obs = {}
            for k, v in traj.items():
                if k in ['actions', 'rewards', 'masks', 'is_success', 'episode_return', 'episode_length', 'env_steps']:
                    # Skip metadata fields
                    continue
                else:
                    # Extract observation at next_t
                    next_obs[k] = v[next_t]
            
            # Calculate n-step discounted reward sum
            # Extract rewards for action_horizon steps
            remaining_rewards = traj['rewards'][t:]
            if len(remaining_rewards) >= action_horizon:
                reward_sequence = remaining_rewards[:action_horizon]
            else:
                # Pad with zeros if not enough remaining rewards
                padding_needed = action_horizon - len(remaining_rewards)
                reward_sequence = np.concatenate([
                    remaining_rewards,
                    np.zeros(padding_needed)
                ])
            
            # # Calculate discounted sum
            # discounts = np.array([discount_factor ** i for i in range(action_horizon)])
            # n_step_reward = np.sum(reward_sequence * discounts)
            
            observations_list.append(obs)
            next_observations_list.append(next_obs)
            actions_list.append(action_sequence_padded)  # Shape: (action_horizon, action_dim)
            rewards_list.append(reward_sequence)  # Shape: (action_horizon,) - reward sequence for chunk RL
            terminals_list.append(1 - traj['masks'][t])  # terminals = 1 - masks
            masks_list.append(bool(traj['masks'][t]))
        
        # Convert lists to arrays with batch dimension as axis 0
        # batch = {}
        
        # Process observations - convert to _model.Observation format
        batch_obs = {}
        batch_next_obs = {}
        
        # Stack all observation fields with batch dimension as axis 0
        for k in observations_list[0].keys():
            batch_obs[k] = np.stack([obs[k] for obs in observations_list], axis=0)
            batch_next_obs[k] = np.stack([next_obs[k] for next_obs in next_observations_list], axis=0)
        
        # Convert to _model.Observation format
        # Reconstruct the nested structure expected by Observation.from_dict
        obs_dict = {
            'image': {
                'base_0_rgb': batch_obs['base_img'],  # [batch_size, 224, 224, 3]
                'left_wrist_0_rgb': batch_obs['wrist_img'],  # [batch_size, 224, 224, 3]
                'right_wrist_0_rgb': np.zeros_like(batch_obs['base_img'])  # [batch_size, 224, 224, 3]
            },
            'image_mask': {
                'base_0_rgb': batch_obs['base_img_mask'],  # [batch_size]
                'left_wrist_0_rgb': batch_obs['wrist_img_mask'],  # [batch_size]
                'right_wrist_0_rgb': np.zeros_like(batch_obs['base_img_mask'], dtype=bool)  # [batch_size]
            },
            'state': batch_obs['state'],  # [batch_size, 7]
            'tokenized_prompt': batch_obs['tokenized_prompt'],  # [batch_size, d]
            'tokenized_prompt_mask': batch_obs['tokenized_prompt_mask']  # [batch_size, d]
        }
        
        next_obs_dict = {
            'image': {
                'base_0_rgb': batch_next_obs['base_img'],  # [batch_size, 224, 224, 3]
                'left_wrist_0_rgb': batch_next_obs['wrist_img'],  # [batch_size, 224, 224, 3]
                'right_wrist_0_rgb': np.zeros_like(batch_next_obs['base_img'])  # [batch_size, 224, 224, 3]
            },
            'image_mask': {
                'base_0_rgb': batch_next_obs['base_img_mask'],  # [batch_size]
                'left_wrist_0_rgb': batch_next_obs['wrist_img_mask'],  # [batch_size]
                'right_wrist_0_rgb': np.zeros_like(batch_next_obs['base_img_mask'], dtype=bool)  # [batch_size]
            },
            'state': batch_next_obs['state'],  # [batch_size, 7]
            'tokenized_prompt': batch_next_obs['tokenized_prompt'],  # [batch_size, d]
            'tokenized_prompt_mask': batch_next_obs['tokenized_prompt_mask']  # [batch_size, d]
        }
        
        # # Convert to JAX arrays and create Observation objects
        obs_dict["image"] = {k: jax.numpy.array(v) for k, v in obs_dict["image"].items()}
        obs_dict["image_mask"] = {k: jax.numpy.array(v) for k, v in obs_dict["image_mask"].items()}
        obs_dict["state"] = jax.numpy.array(obs_dict["state"], dtype=jnp.float32)
        obs_dict["tokenized_prompt"] = jax.numpy.array(obs_dict["tokenized_prompt"], dtype=jnp.int32)
        obs_dict["tokenized_prompt_mask"] = jax.numpy.array(obs_dict["tokenized_prompt_mask"], dtype=jnp.bool_)
        
        next_obs_dict["image"] = {k: jax.numpy.array(v) for k, v in next_obs_dict["image"].items()}
        next_obs_dict["image_mask"] = {k: jax.numpy.array(v) for k, v in next_obs_dict["image_mask"].items()}
        next_obs_dict["state"] = jax.numpy.array(next_obs_dict["state"], dtype=jnp.float32)
        next_obs_dict["tokenized_prompt"] = jax.numpy.array(next_obs_dict["tokenized_prompt"], dtype=jnp.int32)
        next_obs_dict["tokenized_prompt_mask"] = jax.numpy.array(next_obs_dict["tokenized_prompt_mask"], dtype=jnp.bool_)
        

        batch = (
            _model.Observation.from_dict(obs_dict),
            np.stack(actions_list, axis=0).astype(np.float32),
            np.stack(rewards_list, axis=0).astype(np.float32),
            _model.Observation.from_dict(next_obs_dict),
            np.stack(masks_list, axis=0).astype(np.bool_)
        )
        
        return batch

    def get_iterator(self, batch_size: int, keys: Optional[Iterable[str]] = None, indx: Optional[np.ndarray] = None, queue_size: int = 2, action_horizon: int = 50, action_dim: int = 32, discount_factor: float = 0.99):
        """Get an iterator for the buffer data."""
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, keys, indx, action_horizon, action_dim, discount_factor)
                queue.append(data)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def compute_action_stats(self):
        """Compute action statistics for normalization."""
        all_actions = []
        for traj in self.trajectories.values():
            all_actions.append(traj['actions'])
        
        if all_actions:
            all_actions = np.concatenate(all_actions, axis=0)
            return {'mean': all_actions.mean(axis=0), 'std': all_actions.std(axis=0)}
        else:
            return {'mean': np.zeros(self.action_space.shape[0]), 'std': np.ones(self.action_space.shape[0])}

    def normalize_actions(self, action_stats):
        """Normalize actions using provided statistics."""
        # do not normalize gripper dimension (last dimension)
        action_stats = copy.deepcopy(action_stats)
        action_stats['mean'][-1] = 0
        action_stats['std'][-1] = 1
        
        for traj_id, traj in self.trajectories.items():
            normalized_actions = (traj['actions'] - action_stats['mean']) / action_stats['std']
            self.trajectories[traj_id]['actions'] = normalized_actions

    def save(self, filename):
        """Save buffer to file."""
        save_dict = dict(
            trajectories=self.trajectories,
            traj_metadata=self.traj_metadata,
            size=self.size,
            total_steps=self.total_steps,
            _traj_counter=self._traj_counter,
            capacity=self.capacity,
            observation_space=self.observation_space,
            action_space=self.action_space
        )
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f, protocol=4)

    def restore(self, filename):
        """Restore buffer from file."""
        with open(filename, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.trajectories = save_dict['trajectories']
        self.traj_metadata = save_dict['traj_metadata']
        self.size = save_dict['size']
        self.total_steps = save_dict.get('total_steps', 0)  # Backward compatibility
        self._traj_counter = save_dict['_traj_counter']
        self.capacity = save_dict['capacity']
        self.observation_space = save_dict['observation_space']
        self.action_space = save_dict['action_space']