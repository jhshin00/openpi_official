#!/usr/bin/env bash
set -euo pipefail

# 사용할 GPU 번호
export CUDA_VISIBLE_DEVICES=2,3
export HF_LEROBOT_HOME=./datasets
export HF_DATASETS_CACHE=$(pwd)/datasets/cache
export HUGGINGFACE_HUB_CACHE=$(pwd)/datasets//cache/hub

echo "1) UR3 데이터 → LeRobot 포맷으로 변환 시작"
./.venv/bin/python convert_ur3_data_to_lerobot_modified.py --raw_dir=./datasets/ur3_datasets

echo "2) Stats Compute"
./.venv/bin/python scripts/compute_norm_stats.py --config-name=pi0_ur3_lora_finetune

echo "✅ 완료!"