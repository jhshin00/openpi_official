import numpy as np
import pathlib
from tqdm import tqdm
import pathlib

import jax
import flax.nnx as nnx
import wandb
import imageio
import gym
import gym.spaces

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi.transforms import TokenizePrompt
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi_client import image_tools



LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
PALIGEMMA_VOCAB_SIZE = 257_152


def create_expo_obs_space():
    """Create observation space in expo format"""
    return gym.spaces.Dict({
        'image': gym.spaces.Dict({
            'base_0_rgb': gym.spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
            'left_wrist_0_rgb': gym.spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
            'right_wrist_0_rgb': gym.spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
        }),
        'image_mask': gym.spaces.Dict({
            'base_0_rgb': gym.spaces.Box(0, 1, (), dtype=bool),
            'left_wrist_0_rgb': gym.spaces.Box(0, 1, (), dtype=bool),
            'right_wrist_0_rgb': gym.spaces.Box(0, 1, (), dtype=bool),
        }),
        'state': gym.spaces.Box(-np.inf, np.inf, (32,), dtype=np.float32),
        'tokenized_prompt': gym.spaces.Box(0, PALIGEMMA_VOCAB_SIZE, (48,), dtype=np.int32),
        'tokenized_prompt_mask': gym.spaces.Box(0, 1, (48,), dtype=bool),
    })


def create_expo_action_space():
    """Create action space for expo"""
    return gym.spaces.Box(-1, 1, (32,), dtype=np.float32)

def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width)
    return x


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle representation"""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if abs(den) < 1e-6:
        return np.zeros(3)

    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def obs_to_expo_pi0_format_dict(obs, task_description, tokenizer, max_token_len, action_dim=7):
    base_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
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


    quat = np.array(obs["robot0_eef_quat"])
    axis_angle = _quat2axisangle(quat)
    
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            np.array(axis_angle),
            obs["robot0_gripper_qpos"],
        )
    )[None, ...].astype(np.float32)[:, :-1]
    
    # Pad state to match action_dim
    state = pad_to_dim(state, action_dim, axis=-1)
    prompt = str(task_description)
    # tokenizer = _tokenizer.PaligemmaTokenizer(max_token_len)
    obs_dict = TokenizePrompt(tokenizer)({
        "image": images,
        "image_mask": image_masks,
        "state": state,
        "prompt": prompt,
    })
    obs_dict["tokenized_prompt"] = obs_dict["tokenized_prompt"][None, ...]
    obs_dict["tokenized_prompt_mask"] = obs_dict["tokenized_prompt_mask"][None, ...]
    
    return obs_dict

def obs_to_expo_pi0_format(obs, task_description, tokenizer, max_token_len, action_dim=7):
    obs_dict = obs_to_expo_pi0_format_dict(obs, task_description, tokenizer, max_token_len, action_dim)
    
    # Convert numpy arrays to JAX arrays for type compatibility (more efficient)
    obs_dict["image"] = {k: jax.numpy.asarray(v) for k, v in obs_dict["image"].items()}
    obs_dict["image_mask"] = {k: jax.numpy.asarray(v) for k, v in obs_dict["image_mask"].items()}
    obs_dict["state"] = jax.numpy.asarray(obs_dict["state"])
    obs_dict["tokenized_prompt"] = jax.numpy.asarray(obs_dict["tokenized_prompt"])
    obs_dict["tokenized_prompt_mask"] = jax.numpy.asarray(obs_dict["tokenized_prompt_mask"])
    
    return _model.Observation.from_dict(obs_dict)

def obs_dict_to_expo_pi0_format(obs_dict):
    obs_dict["image"] = {k: jax.numpy.array(v) for k, v in obs_dict["image"].items()}
    obs_dict["image_mask"] = {k: jax.numpy.array(v) for k, v in obs_dict["image_mask"].items()}
    obs_dict["state"] = jax.numpy.array(obs_dict["state"])
    obs_dict["tokenized_prompt"] = jax.numpy.array(obs_dict["tokenized_prompt"])
    obs_dict["tokenized_prompt_mask"] = jax.numpy.array(obs_dict["tokenized_prompt_mask"])

    return _model.Observation.from_dict(obs_dict)


def convert_obs_list_to_flat_trajectory(obs_list, action_list, rewards, masks, is_success, episode_length, env_steps):
    """
    Convert list of observations to flat trajectory structure.
    
    Args:
        obs_list: List of observation dictionaries from to_dict()
        action_list: Array of actions
        rewards: Array of rewards
        masks: Array of masks
        is_success: Boolean success flag
        episode_length: Length of episode
        env_steps: Number of environment steps
    
    Returns:
        Flat trajectory dictionary
    """
    traj = {}
    
    # Process images - extract specific keys and concatenate
    image_keys = {
        'base_img': 'base_0_rgb',
        'wrist_img': 'left_wrist_0_rgb'
    }
    
    for traj_key, obs_key in image_keys.items():
        traj[traj_key] = np.concatenate([v["image"][obs_key] for v in obs_list], axis=0)
    
    # Process image masks
    mask_keys = {
        'base_img_mask': 'base_0_rgb',
        'wrist_img_mask': 'left_wrist_0_rgb'
    }
    
    for traj_key, obs_key in mask_keys.items():
        traj[traj_key] = np.concatenate([v["image_mask"][obs_key] for v in obs_list], axis=0)
    
    # Process other observation fields
    other_fields = ['state', 'tokenized_prompt', 'tokenized_prompt_mask']
    for field in other_fields:
        traj[field] = np.concatenate([v[field] for v in obs_list], axis=0)
    
    # Process actions, rewards, masks
    traj["actions"] = action_list
    traj["rewards"] = rewards
    traj["masks"] = masks
    
    # Add metadata
    traj["is_success"] = is_success
    traj["episode_return"] = np.sum(rewards)
    traj["episode_length"] = episode_length
    traj["env_steps"] = env_steps
    
    return traj

def get_libero_env(task, resolution, seed):
    """Initialize LIBERO environment"""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def collect_trajectory_data(rng, config, train_state, env, task_description, tokenizer):
    obs = env.reset()

    obs_list = []
    action_list = []
    rewards = []
    masks = []

    for t in tqdm(range(config.max_timesteps)):
        expo_obs = obs_to_expo_pi0_format_dict(obs, task_description, tokenizer, config.max_token_len, config.action_dim)

        if t % config.action_horizon == 0:
            model = nnx.merge(
                train_state.model_def,
                train_state.critic_params,
                train_state.target_critic_params,
                train_state.actor_params,
                train_state.edit_actor_params,
                train_state.temp_params,
            )
            expo_obs_model_input = obs_dict_to_expo_pi0_format(expo_obs)
            action = model.sample_OTF_actions(rng, expo_obs_model_input).squeeze() # [H, A]

        action_idx = t % config.action_horizon
        curr_action = action[action_idx]

        if len(curr_action) > 7:
            curr_action = curr_action[:7]

        curr_action = np.clip(curr_action, -1.0, 1.0)
        next_obs, reward, done, _ = env.step(curr_action)

        # obs_dict = expo_obs.to_dict()
        obs_list.append(expo_obs)
        action_list.append(curr_action)
        rewards.append(reward)
        masks.append(not done)

        obs = next_obs
        if done:
            break

    obs_dict = obs_to_expo_pi0_format_dict(obs, task_description, tokenizer, config.max_token_len, config.action_dim)
    obs_list.append(obs_dict)
    
    action_list = np.array(action_list)
    rewards = np.array(rewards)
    masks = np.array(masks)

    is_success = (reward == config.env_max_reward)
    episode_length = len(rewards)
    env_steps = t + 1
    
    return obs_list, action_list, rewards, masks, is_success, episode_length, env_steps

def collect_trajectory(rng, config, train_state, env, task_description, tokenizer):
    obs_list, action_list, rewards, masks, is_success, episode_length, env_steps = collect_trajectory_data(
        rng,
        config,
        train_state,
        env,
        task_description,
        tokenizer
    )
    traj = convert_obs_list_to_flat_trajectory(
        obs_list,
        action_list,
        rewards, masks, is_success, episode_length, env_steps)

    return traj

def add_online_data_to_buffer(traj, replay_buffer):
    replay_buffer.insert_traj(traj)

def perform_control_eval(rng, config, train_state, env, task_description, step, tokenizer):
    success_rates = []
    episode_returns = []
    episode_lens = []
    
    logged_success_video = False
    logged_failure_video = False

    for rollout_id in range(config.eval_episodes):
        image_list = []
        rewards = []

        obs = env.reset()
        for t in tqdm(range(config.max_timesteps + config.num_steps_wait), desc="Evaluating trajectory"):
            expo_obs = obs_to_expo_pi0_format_dict(obs, task_description, tokenizer, config.max_token_len, config.action_dim)
            
            if t < config.num_steps_wait:
                obs, reward, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            if (t - config.num_steps_wait) % config.action_horizon == 0:
                model = nnx.merge(
                    train_state.model_def,
                    train_state.critic_params,
                    train_state.target_critic_params,
                    train_state.actor_params,
                    train_state.edit_actor_params,
                    train_state.temp_params,
                )
                expo_obs_model_input = obs_dict_to_expo_pi0_format(expo_obs)
                action = model.sample_OTF_actions(rng, expo_obs_model_input).squeeze() # [H, A]

            action_idx = (t - config.num_steps_wait) % config.action_horizon
            curr_action = action[action_idx]

            if len(curr_action) > 7:
                curr_action = curr_action[:7]

            curr_action = np.clip(curr_action, -1.0, 1.0)
            next_obs, reward, done, _ = env.step(curr_action)

            rewards.append(reward)
            
            # Use the same image processing as in obs_to_expo_pi0_format_dict for consistency
            base_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            base_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(base_img, 224, 224)
            )
            image_list.append(base_img)
            
            obs = next_obs
            if done:
                break

        rewards = np.array(rewards)
        episode_lens.append(t + 1)
        episode_return = np.sum(rewards)
        episode_returns.append(episode_return)
        is_success = (reward == config.env_max_reward)
        success_rates.append(is_success)
        print(f'Rollout Done: {episode_return=}, Success: {is_success}')
        
        if is_success and not logged_success_video:
            video = np.stack(image_list, axis=0)
            video_for_wandb = video.transpose(0, 3, 1, 2)
            wandb.log({f'eval_video/{step}/{task_description}/SUCCESS_{rollout_id}': wandb.Video(video_for_wandb, fps=50)}, step=step)
            logged_success_video = True
        elif not is_success and not logged_failure_video:
            video = np.stack(image_list, axis=0)
            video_for_wandb = video.transpose(0, 3, 1, 2)
            wandb.log({f'eval_video/{step}/{task_description}/FAILURE_{rollout_id}': wandb.Video(video_for_wandb, fps=50)}, step=step)
            logged_failure_video = True

        # # Convert image list to proper video format: (T, H, W, C)
        # video = np.stack(image_list, axis=0)  # Shape: (T, H, W, C)
        # video_for_wandb = video.transpose(0, 3, 1, 2)  # Shape: (T, C, H, W) for wandb
        # print(f'Video shape: {video.shape}')  # Debug print

        # wandb.log({f'eval_video/{step}/{task_description}/{rollout_id}/{is_success}': wandb.Video(video_for_wandb, fps=50)}, step=step)
        
        # Ensure the directory exists before writing the video file
        # video_path = pathlib.Path(config.experiments_dir) / f'eval_video/{task_description}/{rollout_id}.mp4'
        # video_path.parent.mkdir(parents=True, exist_ok=True)
        # imageio.mimwrite(video_path, video, fps=50)

    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)
    wandb.log({'evaluation/avg_return': avg_return}, step=step)
    wandb.log({'evaluation/success_rate': success_rate}, step=step)
    wandb.log({'evaluation/avg_episode_len': avg_episode_len}, step=step)