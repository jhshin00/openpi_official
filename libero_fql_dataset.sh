#!/usr/bin/env bash
set -euo pipefail

# 사용할 GPU 번호
export CUDA_VISIBLE_DEVICES=1,2,3
export HF_LEROBOT_HOME=./datasets
export HF_DATASETS_CACHE=$(pwd)/datasets/cache
export HUGGINGFACE_HUB_CACHE=$(pwd)/datasets//cache/hub

echo "1) Libero 데이터 → LeRobot for FQL 포맷으로 변환 시작"
./.venv/bin/python examples/libero/convert_libero_data_to_lerobot_FQL.py --data_dir=./datasets/libero_raw

echo "2) Stats Compute"
./.venv/bin/python scripts/compute_norm_stats.py --config-name=pi0_fql_libero_lora_finetune

echo "✅ 완료!"