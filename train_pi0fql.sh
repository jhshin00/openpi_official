set -euo pipefail

export CUDA_VISIBLE_DEVICES=1,2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_LEROBOT_HOME=./datasets
export HF_DATASETS_CACHE=$(pwd)/datasets/cache
export HUGGINGFACE_HUB_CACHE=$(pwd)/datasets//cache/hub

./.venv/bin/python scripts/train_fql_test2.py pi0_fql_libero_lora_finetune \
  --exp-name=exp_chunk_qclip_0731_1630 \
  --save-interval=5_000 \
  --keep-period=5_000 \
  --log-interval=100 \
  --batch-size=36 \
  --tau-target=0.005 \
  --rl-loss-coef=0.2 \
  --rl-warmup-steps=10_000 \
  --actor-warmup-steps=5_000 \
  --alpha_q=0.01 \
  --wandb_enabled \
  --resume \
  --no-overwrite \
  --lr-schedule.warmup-steps=1_000 \
  --lr-schedule.peak-lr=2.5e-5 \
  --lr-schedule.decay-lr=2.5e-6 \
  --lr-schedule-critic.warmup-steps=1_000 \
  --lr-schedule-critic.peak-lr=5.0e-5 \
  --lr-schedule-critic.decay-lr=1.0e-6 \
  --critic_updates_per_step=1 \
  --seed=68 \

  # --model.normalize-q-loss \
  # --no-wandb-enabled
  # --resume
  # --no-overwrite
  # --model.no-normalize-q-loss \