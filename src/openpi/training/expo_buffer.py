"""
EXPO Replay Buffer with Libero Offline Data Support (Two-tier, Retrieval-ready)
==============================================================================

- Two-tier pools: OFFLINE (pinned, never evicted) + ONLINE (evicted first)
- Success memory per task (for retrieval indexing)
- Stratified sampling (offline:online) per batch
- Vectorized LIBERO HDF5 -> expo_pi0 trajectory loader (optional)
- No dependency on self.trajectories (removed); pooled storage only

Usage:
    buffer = TrajReplayBuffer(obs_space, action_space, capacity,
                              use_offline_data=True,
                              libero_data_dir="/ssd2/EXPO/datasets/libero_90",
                              offline_dataset_subset_num=300)
    batch = buffer.sample(batch_size=256)  # returns (obs, actions, rewards, next_obs, masks)
    for batch in buffer.get_iterator(batch_size=256): ...

Notes:
- capacity is measured in *steps* (sum of episode_length across all trajectories)
- OFFLINE pool is pinned (never evicted). If capacity is exceeded, ONLINE is evicted first.
"""

from __future__ import annotations

from typing import Iterable, Optional, Dict, Any, List, Tuple
import os
import re
import copy
import pickle
import collections
from collections import deque, defaultdict

import numpy as np
import gym
import gym.spaces
import jax
import jax.numpy as jnp
import h5py

from openpi.data.dataset import Dataset, DatasetDict
from openpi.training.expo_train_utils import pad_to_dim
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.transforms import TokenizePrompt
from openpi_client import image_tools


# ------------------------------
# Helpers for LIBERO conversion
# ------------------------------

def libero_obs_to_expo_pi0_format(libero_obs: Dict[str, np.ndarray],
                                  task_description: str,
                                  max_token_len: int = 48,
                                  action_dim: int = 32) -> Dict[str, Any]:
    """Convert a single step (dict) of libreo obs → expo_pi0 obs (batched length=1)."""
    base_img = np.ascontiguousarray(libero_obs["agentview_rgb"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(libero_obs["eye_in_hand_rgb"][::-1, ::-1])

    base_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(base_img, 224, 224))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))

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

    state = np.concatenate([
        libero_obs["ee_pos"],              # (3,)
        libero_obs["ee_ori"],              # (3,)
        libero_obs["gripper_states"][:1],  # (1,)
    ])[None, ...].astype(np.float32)
    state = pad_to_dim(state, action_dim, axis=-1)

    tokenizer = _tokenizer.PaligemmaTokenizer(max_token_len)
    obs_dict = TokenizePrompt(tokenizer)({
        "image": images,
        "image_mask": image_masks,
        "state": state,
        "prompt": str(task_description),
    })
    obs_dict["tokenized_prompt"] = obs_dict["tokenized_prompt"][None, ...]
    obs_dict["tokenized_prompt_mask"] = obs_dict["tokenized_prompt_mask"][None, ...]
    return obs_dict


def load_libero_trajectory(hdf5_path: str,
                           demo_idx: int = 0,
                           task_description: str = "manipulation task",
                           max_token_len: int = 48,
                           action_dim: int = 32) -> DatasetDict:
    """
    Load one LIBERO demo (vectorized) → flat trajectory dict compatible with insert_traj.
    """
    with h5py.File(hdf5_path, 'r') as f:
        demo = f['data'][f'demo_{demo_idx}']
        actions = demo['actions'][:]                  # (T, 7)
        rewards = demo['rewards'][:]                  # (T,)
        dones = demo['dones'][:]                      # (T,)
        obs_group = demo['obs']

        lib_obs = {k: obs_group[k][:] for k in obs_group.keys()}
        base_imgs = lib_obs["agentview_rgb"]          # (T, 256, 256, 3)
        wrist_imgs = lib_obs["eye_in_hand_rgb"]       # (T, 256, 256, 3)

        base_imgs_flipped = base_imgs[:, ::-1, ::-1, :]
        wrist_imgs_flipped = wrist_imgs[:, ::-1, ::-1, :]

        processed_base_imgs = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(base_imgs_flipped, 224, 224)
        )
        processed_wrist_imgs = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_imgs_flipped, 224, 224)
        )

        images = {
            "base_0_rgb": processed_base_imgs,                    # (T, 224, 224, 3)
            "left_wrist_0_rgb": processed_wrist_imgs,             # (T, 224, 224, 3)
            "right_wrist_0_rgb": np.zeros_like(processed_base_imgs),
        }
        image_masks = {
            "base_0_rgb": np.ones(len(actions), dtype=bool),      # (T,)
            "left_wrist_0_rgb": np.ones(len(actions), dtype=bool),
            "right_wrist_0_rgb": np.zeros(len(actions), dtype=bool),
        }

        state = np.concatenate([
            lib_obs["ee_pos"],                    # (T,3)
            lib_obs["ee_ori"],                    # (T,3)
            lib_obs["gripper_states"][:, :1],     # (T,1)
        ], axis=1).astype(np.float32)             # (T,7)
        state = pad_to_dim(state, action_dim, axis=-1)

        tokenizer = _tokenizer.PaligemmaTokenizer(max_token_len)
        tokenized = TokenizePrompt(tokenizer)({"prompt": str(task_description)})
        tok = tokenized["tokenized_prompt"]                # (1,L)
        tok_m = tokenized["tokenized_prompt_mask"]         # (1,L)
        tokenized_prompts = np.tile(tok,   (len(actions), 1))
        tokenized_masks   = np.tile(tok_m, (len(actions), 1))

        masks = (1 - dones).astype(np.bool_)
        is_success = bool(rewards[-1] == 1.0)
        episode_length = int(len(actions))
        env_steps = episode_length

        print(f"  Loaded trajectory: {task_description}")
        print(f"    - Episode length: {episode_length}")
        print(f"    - Success: {is_success}")
        print(f"    - Episode return: {rewards.sum():.1f}")

        traj = {
            'task_name': task_description,
            'base_img': images['base_0_rgb'],                       # (T, 224, 224, 3)
            'wrist_img': images['left_wrist_0_rgb'],                # (T, 224, 224, 3)
            'base_img_mask': image_masks['base_0_rgb'],             # (T,)
            'wrist_img_mask': image_masks['left_wrist_0_rgb'],      # (T,)
            'state': state,                                         # (T, action_dim>=7)
            'tokenized_prompt': tokenized_prompts,                  # (T, L)
            'tokenized_prompt_mask': tokenized_masks,               # (T, L)
            'actions': actions,                                     # (T, 7)
            'rewards': rewards,                                     # (T,)
            'masks': masks,                                         # (T,)
            'is_success': is_success,
            'episode_return': float(rewards.sum()),
            'episode_length': episode_length,
            'env_steps': env_steps,
        }
        return traj


def load_sample_libero_trajectories(dataset_dir: str,
                                    max_trajectories: Optional[int] = None,
                                    max_token_len: int = 48,
                                    action_dim: int = 32) -> List[DatasetDict]:
    """Load a subset of LIBERO trajectories from directory of *.hdf5 files."""
    trajectories: List[DatasetDict] = []
    hdf5_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')])
    print(f"Found {len(hdf5_files)} HDF5 files. Loading trajectories...")

    for file_idx, hdf5_file in enumerate(hdf5_files):
        if max_trajectories is not None and len(trajectories) >= max_trajectories:
            break

        hdf5_path = os.path.join(dataset_dir, hdf5_file)
        task_name = hdf5_file.replace('_demo.hdf5', '')
        task_desc = re.sub(r'^[A-Z0-9_]+_', '', task_name).replace('_', ' ').lower()

        print(f"\nProcessing file {file_idx + 1}/{len(hdf5_files)}: {hdf5_file}")
        print(f"  Task: {task_desc}")

        with h5py.File(hdf5_path, 'r') as f:
            demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
            demo_indices = sorted(int(k.split('_')[1]) for k in demo_keys)

            if max_trajectories is not None:
                remaining = max_trajectories - len(trajectories)
                remaining_files = len(hdf5_files) - file_idx
                max_demos_per_file = max(1, remaining // max(1, remaining_files))
                max_demos_per_file = min(max_demos_per_file, len(demo_indices))
            else:
                max_demos_per_file = len(demo_indices)

            print(f"  Loading {max_demos_per_file} demos from this file")

            for demo_idx in demo_indices[:max_demos_per_file]:
                if max_trajectories is not None and len(trajectories) >= max_trajectories:
                    break
                try:
                    traj = load_libero_trajectory(hdf5_path, demo_idx, task_desc,
                                                  max_token_len, action_dim)
                    traj["task_name"] = task_desc
                    trajectories.append(traj)
                    print(f"  ✓ Total loaded: {len(trajectories)}/{max_trajectories or 'all'}")
                except Exception as e:
                    print(f"  ✗ Warning: Failed {hdf5_file} demo_{demo_idx}: {e}")
                    continue

    print(f"Loaded {len(trajectories)} trajectories from {len(hdf5_files)} files")
    return trajectories


def load_all_libero_trajectories(dataset_dir: str,
                                 max_token_len: int = 48,
                                 action_dim: int = 32) -> List[DatasetDict]:
    """Load all LIBERO trajectories."""
    trajectories: List[DatasetDict] = []
    hdf5_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')])

    for hdf5_file in hdf5_files:
        hdf5_path = os.path.join(dataset_dir, hdf5_file)
        task_name = hdf5_file.replace('_demo.hdf5', '')
        task_desc = re.sub(r'^[A-Z0-9_]+_', '', task_name).replace('_', ' ').lower()

        with h5py.File(hdf5_path, 'r') as f:
            demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
            demo_indices = sorted(int(k.split('_')[1]) for k in demo_keys)

            for demo_idx in demo_indices:
                try:
                    traj = load_libero_trajectory(hdf5_path, demo_idx, task_desc,
                                                  max_token_len, action_dim)
                    traj["task_name"] = task_desc
                    trajectories.append(traj)
                except Exception as e:
                    print(f"Warning: Failed {hdf5_file} demo_{demo_idx}: {e}")
                    continue

    print(f"Loaded {len(trajectories)} trajectories from {len(hdf5_files)} files")
    return trajectories


# ------------------------------
# Replay Buffer (two-tier)
# ------------------------------

class TrajReplayBuffer(Dataset):
    """
    Trajectory-based replay buffer (two-tier: offline pinned + online evictable).
    """

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 capacity: int,
                 use_offline_data: bool,
                 libero_data_dir: str = "/ssd2/EXPO/datasets/libero_goal",
                 offline_dataset_subset_num: Optional[int] = None,
                 batch_offline_ratio: float = 0.5,
                 success_memory_per_task: int = 10,
                 eviction: str = "fifo"):
        self.observation_space = observation_space
        self.action_space = action_space
        self.capacity = int(capacity)  # in steps
        self.use_offline_data = bool(use_offline_data)
        self.libero_data_dir = str(libero_data_dir)
        self.offline_dataset_subset_num = offline_dataset_subset_num

        print("making trajectory replay buffer of capacity ", self.capacity, "steps")

        # Pooled storage
        # item: {"data": traj_dict, "task": str, "is_success": bool, "pin": bool}
        self.offline_pool: Dict[str, Dict[str, Any]] = {}
        self.online_pool:  Dict[str, Dict[str, Any]] = {}
        self._offline_q: deque[str] = deque()
        self._online_q:  deque[str] = deque()

        self.offline_size: int = 0  # in steps
        self.online_size: int = 0

        self.success_memory: Dict[str, deque[str]] = defaultdict(
            lambda: deque(maxlen=int(success_memory_per_task))
        )
        self.batch_offline_ratio = float(batch_offline_ratio)
        self.eviction = str(eviction)

        self.size = 0            # number of trajectories
        self.total_steps = 0     # total steps across all trajectories
        self._traj_counter = 0
        self.streaming_buffer_size = None

        if self.use_offline_data:
            print("Loading offline libero data...")
            self._load_offline_data()

    # -------- Offline loading --------

    def _load_offline_data(self):
        try:
            if self.offline_dataset_subset_num is not None:
                print(f"\n=== Loading {self.offline_dataset_subset_num} trajectories from LIBERO ===")
                trajectories = load_sample_libero_trajectories(
                    self.libero_data_dir,
                    max_trajectories=int(self.offline_dataset_subset_num),
                    max_token_len=48,
                    action_dim=32
                )
            else:
                print("\n=== Loading all trajectories from LIBERO ===")
                trajectories = load_all_libero_trajectories(
                    self.libero_data_dir, max_token_len=48, action_dim=32
                )

            print(f"\n=== Inserting {len(trajectories)} trajectories into buffer ===")
            loaded_count = 0
            for traj_idx, traj in enumerate(trajectories):
                try:
                    if self.total_steps + int(traj['episode_length']) > self.capacity:
                        print(f"Buffer capacity reached. Loaded {loaded_count} trajectories ({self.total_steps} steps).")
                        break

                    task_name = traj.get("task_name", "unknown")
                    is_success = bool(traj.get("is_success", True))
                    self.insert_traj(traj, source="offline", task=task_name, is_success=is_success)
                    loaded_count += 1

                    if loaded_count % 10 == 0:
                        print(f"  Inserted {loaded_count}/{len(trajectories)} trajectories "
                              f"({self.total_steps} total steps)")

                except Exception as e:
                    print(f"  ✗ Warning: Failed to insert trajectory {traj_idx}: {e}")
                    continue

            print(f"\n=== Successfully loaded {loaded_count} trajectories into buffer ===")
            print(f"Buffer size: {self.size} trajectories, {self.total_steps} total steps")
            util = 100.0 * self.total_steps / max(1, self.capacity)
            print(f"Buffer utilization: {util:.1f}%")

        except Exception as e:
            print(f"Error loading offline data: {e}")
            print("Continuing with empty buffer...")

    # -------- Basic stats --------

    def __len__(self) -> int:
        return self.size

    def length(self) -> int:
        return self.size

    # -------- Insert / Evict --------

    def insert_traj(self, traj: DatasetDict, *,
                    source: str = "online",
                    task: Optional[str] = "unknown",
                    is_success: bool = True):
        """
        Insert a complete trajectory.
        """
        traj_length = int(traj.get('episode_length', len(traj['actions'])))
        if 'traj_id' not in traj:
            traj['traj_id'] = f"traj_{self._traj_counter}"
        traj_id = str(traj['traj_id'])

        if task is None:
            task = traj.get('task_name', 'unknown')

        item = dict(data=traj, task=str(task), is_success=bool(is_success))
        if source == "offline":
            item["pin"] = True
            self.offline_pool[traj_id] = item
            self._offline_q.append(traj_id)
            self.offline_size += traj_length
        else:
            item["pin"] = False
            self.online_pool[traj_id] = item
            self._online_q.append(traj_id)
            self.online_size += traj_length

        if bool(is_success):
            self.success_memory[str(task)].append(traj_id)

        self.size += 1
        self.total_steps += traj_length
        self._traj_counter += 1

        self._rebalance_after_insert()

    def _rebalance_after_insert(self):
        # Evict ONLY from ONLINE if capacity exceeded
        while (self.offline_size + self.online_size) > self.capacity:
            self._evict_from_online()

    def _evict_from_online(self):
        if not self._online_q:
            print("[WARN] capacity exceeded but no online traj to evict; consider increasing capacity.")
            return
        if self.eviction == "fifo":
            victim = self._online_q.popleft()
        else:
            # priority policy can be added here (e.g., TD-error)
            victim = self._online_q.popleft()

        item = self.online_pool.pop(victim, None)
        if item is not None:
            T = int(item["data"].get("episode_length", len(item["data"]["actions"])))
            self.online_size -= T
            self.size = max(0, self.size - 1)
            self.total_steps = max(0, self.total_steps - T)

    # -------- Sampling / Iterator --------

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None,
               action_horizon: int = 50,
               action_dim: int = 32,
               discount_factor: float = 0.99):
        """
        Stratified sampling with action horizon chunking.

        Returns:
            (obs: Observation, actions: [B,H,A], rewards: [B,H], next_obs: Observation, masks: [B])
        """
        if (len(self.offline_pool) + len(self.online_pool)) == 0:
            return None

        # 1) choose trajectories (offline:online ratio)
        k_off = int(round(self.batch_offline_ratio * batch_size))
        k_on = batch_size - k_off

        off_ids = list(self.offline_pool.keys())
        on_ids  = list(self.online_pool.keys())

        sel_off, sel_on = [], []
        if k_off > 0 and len(off_ids) > 0:
            sel_off = np.random.choice(off_ids, size=min(k_off, len(off_ids)),
                                       replace=(len(off_ids) < k_off))
        if k_on > 0 and len(on_ids) > 0:
            sel_on = np.random.choice(on_ids, size=min(k_on, len(on_ids)),
                                      replace=(len(on_ids) < k_on))

        need = batch_size - (len(sel_off) + len(sel_on))
        if need > 0:
            remain = list(set(on_ids) - set(sel_on)) if len(on_ids) > 0 else []
            if len(remain) > 0:
                extra = np.random.choice(remain, size=min(need, len(remain)),
                                         replace=(len(remain) < need))
                sel_on = list(sel_on) + list(extra)
                need = batch_size - (len(sel_off) + len(sel_on))
            if need > 0 and len(off_ids) > 0:
                remain = list(set(off_ids) - set(sel_off))
                if len(remain) > 0:
                    extra = np.random.choice(remain, size=min(need, len(remain)),
                                             replace=(len(remain) < need))
                    sel_off = list(sel_off) + list(extra)

        selected_traj_ids = list(sel_off) + list(sel_on)

        # 2) build chunked batch
        observations_list, next_observations_list = [], []
        actions_list, rewards_list, masks_list = [], [], []

        for traj_id in selected_traj_ids:
            item = self.offline_pool.get(traj_id) or self.online_pool.get(traj_id)
            assert item is not None, f"missing traj {traj_id}"
            traj = item["data"]

            T = int(traj['episode_length'])
            t = np.random.randint(0, T) if T > 1 else 0

            # single-step obs at t
            obs_t = {
                'base_img': traj['base_img'][t],
                'wrist_img': traj['wrist_img'][t],
                'base_img_mask': traj['base_img_mask'][t],
                'wrist_img_mask': traj['wrist_img_mask'][t],
                'state': traj['state'][t],
                'tokenized_prompt': traj['tokenized_prompt'][t],
                'tokenized_prompt_mask': traj['tokenized_prompt_mask'][t],
            }

            # H-step action chunk (pad with last)
            rem_actions = traj['actions'][t:]
            if len(rem_actions) >= action_horizon:
                act_seq = rem_actions[:action_horizon]
            else:
                last_action = rem_actions[-1:] if len(rem_actions) > 0 else traj['actions'][-1:]
                pad_n = action_horizon - len(rem_actions)
                act_seq = np.concatenate([rem_actions, np.tile(last_action, (pad_n, 1))], axis=0)

            # pad action from 7D → action_dim (keep original 7 dims; rest zeros)
            if action_dim > act_seq.shape[1]:
                pad = np.zeros((action_horizon, action_dim - act_seq.shape[1]), dtype=act_seq.dtype)
                act_seq = np.concatenate([act_seq, pad], axis=1)

            # next obs at t+H (clamped)
            next_t = min(t + action_horizon, T - 1)
            next_obs_t = {
                'base_img': traj['base_img'][next_t],
                'wrist_img': traj['wrist_img'][next_t],
                'base_img_mask': traj['base_img_mask'][next_t],
                'wrist_img_mask': traj['wrist_img_mask'][next_t],
                'state': traj['state'][next_t],
                'tokenized_prompt': traj['tokenized_prompt'][next_t],
                'tokenized_prompt_mask': traj['tokenized_prompt_mask'][next_t],
            }

            # H rewards (pad zeros)
            rem_rewards = traj['rewards'][t:]
            if len(rem_rewards) >= action_horizon:
                rew_seq = rem_rewards[:action_horizon]
            else:
                pad_n = action_horizon - len(rem_rewards)
                rew_seq = np.concatenate([rem_rewards, np.zeros(pad_n, dtype=rem_rewards.dtype)], axis=0)

            observations_list.append(obs_t)
            next_observations_list.append(next_obs_t)
            actions_list.append(act_seq.astype(np.float32))
            rewards_list.append(rew_seq.astype(np.float32))
            masks_list.append(bool(traj['masks'][t]))

        # 3) stack into Observation batch
        def stack_field(lst, key):
            return np.stack([e[key] for e in lst], axis=0)

        batch_obs = {k: stack_field(observations_list, k) for k in observations_list[0].keys()}
        batch_next_obs = {k: stack_field(next_observations_list, k) for k in next_observations_list[0].keys()}

        obs_dict = {
            'image': {
                'base_0_rgb': batch_obs['base_img'],              # [B, 224, 224, 3]
                'left_wrist_0_rgb': batch_obs['wrist_img'],
                'right_wrist_0_rgb': np.zeros_like(batch_obs['base_img']),
            },
            'image_mask': {
                'base_0_rgb': batch_obs['base_img_mask'],         # [B]
                'left_wrist_0_rgb': batch_obs['wrist_img_mask'],  # [B]
                'right_wrist_0_rgb': np.zeros_like(batch_obs['base_img_mask'], dtype=bool),
            },
            'state': batch_obs['state'],                          # [B, S]
            'tokenized_prompt': batch_obs['tokenized_prompt'],    # [B, L]
            'tokenized_prompt_mask': batch_obs['tokenized_prompt_mask'],
        }

        next_obs_dict = {
            'image': {
                'base_0_rgb': batch_next_obs['base_img'],
                'left_wrist_0_rgb': batch_next_obs['wrist_img'],
                'right_wrist_0_rgb': np.zeros_like(batch_next_obs['base_img']),
            },
            'image_mask': {
                'base_0_rgb': batch_next_obs['base_img_mask'],
                'left_wrist_0_rgb': batch_next_obs['wrist_img_mask'],
                'right_wrist_0_rgb': np.zeros_like(batch_next_obs['base_img_mask'], dtype=bool),
            },
            'state': batch_next_obs['state'],
            'tokenized_prompt': batch_next_obs['tokenized_prompt'],
            'tokenized_prompt_mask': batch_next_obs['tokenized_prompt_mask'],
        }

        # to jax arrays
        def to_jax_obs(d):
            d["image"] = {k: jax.numpy.array(v) for k, v in d["image"].items()}
            d["image_mask"] = {k: jax.numpy.array(v) for k, v in d["image_mask"].items()}
            d["state"] = jax.numpy.array(d["state"], dtype=jnp.float32)
            d["tokenized_prompt"] = jax.numpy.array(d["tokenized_prompt"], dtype=jnp.int32)
            d["tokenized_prompt_mask"] = jax.numpy.array(d["tokenized_prompt_mask"], dtype=jnp.bool_)
            return d

        obs_o = _model.Observation.from_dict(to_jax_obs(obs_dict))
        nxt_o = _model.Observation.from_dict(to_jax_obs(next_obs_dict))

        batch = (
            obs_o,
            np.stack(actions_list, axis=0).astype(np.float32),   # [B,H,A]
            np.stack(rewards_list, axis=0).astype(np.float32),   # [B,H]
            nxt_o,
            np.stack(masks_list, axis=0).astype(np.bool_),       # [B]
        )
        return batch

    def get_iterator(self,
                     batch_size: int,
                     keys: Optional[Iterable[str]] = None,
                     indx: Optional[np.ndarray] = None,
                     queue_size: int = 2,
                     action_horizon: int = 50,
                     action_dim: int = 32,
                     discount_factor: float = 0.99):
        """Prefetching iterator (deque-based)."""
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, keys, indx, action_horizon, action_dim, discount_factor)
                queue.append(data)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    # -------- Stats / Normalization --------

    def compute_action_stats(self) -> Dict[str, np.ndarray]:
        all_actions = []
        for it in self.offline_pool.values():
            all_actions.append(it["data"]["actions"])
        for it in self.online_pool.values():
            all_actions.append(it["data"]["actions"])
        if all_actions:
            A = np.concatenate(all_actions, axis=0)
            return {'mean': A.mean(axis=0), 'std': A.std(axis=0)}
        else:
            return {'mean': np.zeros(self.action_space.shape[0]),
                    'std': np.ones(self.action_space.shape[0])}

    def normalize_actions(self, action_stats: Dict[str, np.ndarray]):
        # do not normalize gripper dimension (last dim)
        stats = copy.deepcopy(action_stats)
        stats['mean'][-1] = 0
        stats['std'][-1] = 1
        eps = 1e-8
        for pool in (self.offline_pool, self.online_pool):
            for it in pool.values():
                A = it["data"]["actions"]
                it["data"]["actions"] = (A - stats['mean']) / (stats['std'] + eps)

    # -------- Save / Restore --------

    def save(self, filename: str):
        save_dict = dict(
            offline_pool=self.offline_pool,
            online_pool=self.online_pool,
            offline_q=list(self._offline_q),
            online_q=list(self._online_q),
            offline_size=self.offline_size,
            online_size=self.online_size,
            size=self.size,
            total_steps=self.total_steps,
            _traj_counter=self._traj_counter,
            capacity=self.capacity,
            observation_space=self.observation_space,
            action_space=self.action_space,
            batch_offline_ratio=self.batch_offline_ratio,
            eviction=self.eviction,
        )
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f, protocol=4)

    def restore(self, filename: str):
        with open(filename, 'rb') as f:
            d = pickle.load(f)
        self.offline_pool = d['offline_pool']
        self.online_pool  = d['online_pool']
        self._offline_q = deque(d['offline_q'])
        self._online_q  = deque(d['online_q'])
        self.offline_size = d['offline_size']
        self.online_size  = d['online_size']
        self.size = d['size']
        self.total_steps = d['total_steps']
        self._traj_counter = d['_traj_counter']
        self.capacity = d['capacity']
        self.observation_space = d['observation_space']
        self.action_space = d['action_space']
        self.batch_offline_ratio = d.get('batch_offline_ratio', self.batch_offline_ratio)
        self.eviction = d.get('eviction', self.eviction)

    # -------- Convenience (optional) --------

    def get_traj(self, traj_id: str) -> Optional[DatasetDict]:
        it = self.offline_pool.get(traj_id) or self.online_pool.get(traj_id)
        return None if it is None else it["data"]

    def iter_trajs(self):
        for it in self.offline_pool.values():
            yield it["data"]
        for it in self.online_pool.values():
            yield it["data"]

    def stats(self) -> Dict[str, Any]:
        return {
            "offline_trajs": len(self.offline_pool),
            "online_trajs": len(self.online_pool),
            "offline_size": self.offline_size,
            "online_size": self.online_size,
            "total_steps": self.total_steps,
            "num_trajs": self.size,
        }