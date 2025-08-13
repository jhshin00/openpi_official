#!/usr/bin/env bash
set -euo pipefail

# 사용할 GPU 번호
export CUDA_VISIBLE_DEVICES=2,3
export HF_LEROBOT_HOME=./datasets
export HF_DATASETS_CACHE=$(pwd)/datasets/cache
export HUGGINGFACE_HUB_CACHE=$(pwd)/datasets//cache/hub

echo "Stats Compute"
./.venv/bin/python scripts/compute_norm_stats.py --config-name=pi0_ur3_lora_finetune

echo "✅ 완료!"