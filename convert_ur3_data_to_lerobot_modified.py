"""
UR3 데이터를 LeRobot 형식으로 변환하는 스크립트

현재 HDF5 구조:
data.hdf5
├── data
│   ├── actions              # (n_frames, 7)
│   ├── base_rgb             # (n_frames, 224, 224, 3)
│   ├── ee_pos_quat          # (n_frames, 7)
│   ├── gripper_position     # (n_frames, 1)
│   ├── joint_positions      # (n_frames, 7)
│   └── wrist_rgb            # (n_frames, 224, 224, 3)

LeRobot 형식으로 변환:
observation.state (joint_positions)
observation.image.base_image (base_rgb)
observation.image.wrist_image (wrist_rgb)
observation.ee_pos_quat (ee_pos_quat)
observation.gripper_position (gripper_position)
actions

Usage:
uv run convert_ur3_data_to_lerobot_modified.py --raw_dir /path/to/ur3_datasets --repo_id ur3_dataset

If you want to push your dataset to the Hugging Face Hub:
uv run convert_ur3_data_to_lerobot_modified.py --raw_dir /path/to/ur3_datasets --repo_id ur3_dataset --push_to_hub
"""

import shutil
from pathlib import Path
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro

REPO_NAME = "ur3_dataset_modified"  # 기본 출력 데이터셋 이름


def main(
    raw_dir: Path,
    repo_id: str = REPO_NAME,
    *,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "image",
):
    # 출력 디렉토리 정리
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    # HDF5 파일들 찾기
    hdf5_files = sorted(raw_dir.glob("**/data.hdf5"))
    
    if not hdf5_files:
        print(f"No data.hdf5 files found in {raw_dir}")
        return
    
    print(f"Found {len(hdf5_files)} data.hdf5 files")

    # LeRobot 데이터셋 생성
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="UR3",
        fps=30,  # 확인된 FPS
        features={
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "base_image": {
                "dtype": mode,
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": mode,
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "ee_pos_quat": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["ee_pos_quat"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 각 HDF5 파일을 에피소드로 변환
    for hdf5_file in tqdm.tqdm(hdf5_files, desc="Converting episodes"):
        # 파일 경로에서 task 정보 추출
        path_parts = hdf5_file.parts
        task = "ur3_task"  # 기본값
        
        for part in path_parts:
            if part.startswith("TASK"):
                # TASK 숫자와 언더스코어 제거, 공백으로 변환
                # 예: TASK_4_pick_up_the_pink_cup -> pick up the pink cup
                task = part
                # TASK 부분 완전히 제거 (TASK, TASK_4, TASK1_ 등)
                if "_" in task:
                    # 첫 번째 언더스코어 이후 부분만 가져오기
                    task = task.split("_", 1)[1]
                    # 만약 숫자로 시작한다면 그 숫자도 제거
                    if task and task[0].isdigit():
                        # 숫자 부분을 건너뛰고 다음 언더스코어 이후부터
                        if "_" in task:
                            task = task.split("_", 1)[1]
                        else:
                            task = ""  # 숫자만 있다면 빈 문자열
                # 나머지 언더스코어를 공백으로 변환
                task = task.replace("_", " ")
                break

        # HDF5 파일에서 데이터 로드
        with h5py.File(hdf5_file, "r") as f:
            data_group = f["data"]
            
            # 각 프레임을 데이터셋에 추가
            num_frames = data_group["joint_positions"].shape[0]
            
            for i in range(num_frames):
                frame = {
                    "state": torch.from_numpy(data_group["joint_positions"][i]).float(),
                    "base_image": data_group["base_rgb"][i][:, :, ::-1],
                    "wrist_image": data_group["wrist_rgb"][i][:, :, ::-1],
                    "ee_pos_quat": torch.from_numpy(data_group["ee_pos_quat"][i]).float(),
                    "gripper_position": torch.from_numpy(data_group["gripper_position"][i]).float(),
                    "actions": torch.from_numpy(data_group["actions"][i]).float(),
                    "task": task,
                }
                
                dataset.add_frame(frame)
            
            # 에피소드 저장
            dataset.save_episode()

    # Hugging Face Hub에 업로드 (선택사항)
    if push_to_hub:
        dataset.push_to_hub(
            tags=["ur3", "robot", "manipulation"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

    print(f"Conversion completed! Dataset saved to {output_path}")


if __name__ == "__main__":
    tyro.cli(main) 