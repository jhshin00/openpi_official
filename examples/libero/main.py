import collections
import dataclasses
import logging
import math
import pathlib
import pickle

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    
    # Local mode - load policy directly instead of using websocket
    use_local_policy: bool = True
    checkpoint_config: str = "pi0_libero"  # e.g., "pi0_libero"
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi0_libero"  # e.g., "gs://openpi-assets/checkpoints/pi0_libero"

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_object"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 10  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    save_embeddings: bool = True  # Whether to save VLM embeddings for visualization
    embeddings_out_path: str = "data/libero/embeddings/libero_finetune"  # Path to save embeddings

    seed: int = 42  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # Create policy client
    if args.use_local_policy:
        if args.save_embeddings:
            # Import here to avoid dependency issues when not using local mode
            from openpi.policies import policy_config as _policy_config
            from openpi.training import config as _config
            
            if args.checkpoint_config is None or args.checkpoint_dir is None:
                raise ValueError("checkpoint_config and checkpoint_dir must be provided when use_local_policy=True")
            
            config = _config.get_config(args.checkpoint_config)
            client = _policy_config.create_trained_embedding_policy(
                config, args.checkpoint_dir, default_prompt=None
            )
            logging.info("Loaded local embedding policy from checkpoint")
        else:
            from openpi.policies import policy_config as _policy_config
            from openpi.training import config as _config
            
            if args.checkpoint_config is None or args.checkpoint_dir is None:
                raise ValueError("checkpoint_config and checkpoint_dir must be provided when use_local_policy=True")
            
            config = _config.get_config(args.checkpoint_config)
            client = _policy_config.create_trained_policy(
                config, args.checkpoint_dir, default_prompt=None
            )
            logging.info("Loaded local policy from checkpoint")
    else:
        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Prepare embedding storage if needed
    if args.save_embeddings:
        pathlib.Path(args.embeddings_out_path).mkdir(parents=True, exist_ok=True)
        all_embeddings = []

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            
            # For embedding collection
            if args.save_embeddings:
                episode_embeddings = []
                episode_metadata = {
                    "task_id": task_id,
                    "task_description": task_description,
                    "episode_idx": episode_idx,
                }

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict (needed for both action inference and embedding extraction)
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(task_description),
                    }

                    # Collect embeddings at EVERY step if enabled
                    if args.save_embeddings and args.use_local_policy:
                        # Extract embedding only (without action inference)
                        if hasattr(client, 'extract_embedding'):
                            emb_result = client.extract_embedding(element)
                            vlm_emb = emb_result["vlm_embedding"]  # (seq_len, emb_dim)
                            vlm_mask = emb_result["vlm_mask"]  # (seq_len,)
                            
                            # Mask and average
                            masked_emb = vlm_emb * vlm_mask[:, None]
                            avg_emb = masked_emb.sum(axis=0) / (vlm_mask.sum() + 1e-8)
                            
                            episode_embeddings.append({
                                "timestep": t,
                                "embedding": avg_emb,
                            })

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Query model to get action
                        result = client.infer(element)
                        action_chunk = result["actions"]
                        
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save embeddings if enabled
            if args.save_embeddings and episode_embeddings:
                episode_metadata["success"] = done
                episode_metadata["num_timesteps"] = t
                all_embeddings.append({
                    "metadata": episode_metadata,
                    "embeddings": episode_embeddings,
                })

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            ### TODO jhshin video save x
            # imageio.mimwrite(
            #     pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
            #     [np.asarray(x) for x in replay_images],
            #     fps=10,
            # )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    
    # Save all embeddings to file
    if args.save_embeddings:
        embeddings_path = pathlib.Path(args.embeddings_out_path) / f"embeddings_{args.task_suite_name}.pkl"
        with open(embeddings_path, "wb") as f:
            pickle.dump(all_embeddings, f)
        logging.info(f"Saved embeddings to {embeddings_path}")
        logging.info(f"Total episodes with embeddings: {len(all_embeddings)}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
