set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_LEROBOT_HOME=./datasets
export HF_DATASETS_CACHE=$(pwd)/datasets/cache
export HUGGINGFACE_HUB_CACHE=$(pwd)/datasets//cache/hub

echo "UR3 Policy Serving"

./.venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config   pi0_ur3_lora_finetune \
  --policy.dir      ./checkpoints/pi0_ur3_lora_finetune/exp1/8999 \