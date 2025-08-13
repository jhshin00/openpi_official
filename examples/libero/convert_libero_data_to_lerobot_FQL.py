"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
import numpy as np
from pathlib import Path

REPO_NAME = "libero_fql"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    # output_path = Path("./datasets/libero_fql")
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
            "reward": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["reward"],
            },
            "next_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "next_wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "next_state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "terminal": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["terminal"],
            },
        },
        root="./datasets/libero_fql",
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            prev_obs = None
            prev_action = None
            prev_reward = None
            prev_terminal = None
            prev_lang = None

            for step in episode["steps"].as_numpy_iterator():
                obs = step["observation"]
                action = step["action"]
                reward = step["reward"]
                terminal = step["is_terminal"]
                lang = step["language_instruction"].decode()

                if prev_obs is not None:
                    dataset.add_frame(
                        {
                            "image": prev_obs["image"],
                            "wrist_image": prev_obs["wrist_image"],
                            "state": prev_obs["state"],
                            "actions": prev_action,
                            "reward": np.array([prev_reward], dtype=np.float32),
                            "terminal": np.array([prev_terminal], dtype=bool),
                            "next_image": obs["image"],
                            "next_wrist_image": obs["wrist_image"],
                            "next_state": obs["state"],
                            "task": prev_lang,
                        }
                    )
                prev_obs = obs
                prev_action = action
                prev_reward = reward
                prev_terminal = terminal
                prev_lang = lang
            
            if prev_obs is not None:
                dataset.add_frame(
                    {
                        "image": prev_obs["image"],
                        "wrist_image": prev_obs["wrist_image"],
                        "state": prev_obs["state"],
                        "actions": prev_action,
                        "reward": np.array([prev_reward], dtype=np.float32),
                        "terminal": np.array([prev_terminal], dtype=bool),
                        "next_image": np.zeros_like(prev_obs["image"]),
                        "next_wrist_image": np.zeros_like(prev_obs["wrist_image"]),
                        "next_state": np.zeros_like(prev_obs["state"]),
                        "task": prev_lang,
                    }
                )

            dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
