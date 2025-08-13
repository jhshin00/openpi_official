set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_LEROBOT_HOME=./datasets
export HF_DATASETS_CACHE=$(pwd)/datasets/cache
export HUGGINGFACE_HUB_CACHE=$(pwd)/datasets//cache/hub

./.venv/bin/python scripts/train.py pi0_ur3_lora_finetune \
  --exp-name=exp1 \
  --num_train_steps=9_000 \
  --save-interval=1_000 \
  --keep-period=3_000 \
  --log-interval=10 \
  --batch-size=16 \
  --wandb_enabled \
  --no-resume \
  --overwrite \
  --lr-schedule.warmup-steps=750 \
  --lr-schedule.peak-lr=2.5e-5 \
  --lr-schedule.decay-lr=2.5e-6 \
  --seed=42 \
#   --exp-name=exp1 \
#   --num_train_steps=30_000 \
#   --save-interval=2_500 \
#   --keep-period=5_000 \
#   --log-interval=10 \
#   --batch-size=32 \
#   --wandb_enabled \
#   --no-resume \
#   --overwrite \
#   --lr-schedule.warmup-steps=1_000 \
#   --lr-schedule.peak-lr=2.5e-5 \
#   --lr-schedule.decay-lr=2.5e-6 \
#   --seed=42 \

