set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# export HF_LEROBOT_HOME=./datasets
# export HF_DATASETS_CACHE=$(pwd)/datasets/cache
# export HUGGINGFACE_HUB_CACHE=$(pwd)/datasets//cache/hub


# export JAX_PLATFORM_NAME=cuda
# export JAX_PJRT_USE_C_API_ON_GPU=true

# # NCCL (단일 호스트 2GPU에서 무난한 설정)
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_SHM_DISABLE=1
# export NCCL_COLLNET_ENABLE=0
# export NCCL_SOCKET_IFNAME=enp70s0
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=WARN


python scripts/train_pi0_expo.py pi0_expo_libero \
  --exp-name=exp5 \
  --num-train-steps=50_000 \
  --batch-size=32 \
  --seed=64 \
  --log-interval=10 \
  --save-interval=2500 \
  --keep-period=5000 \
  --overwrite \
  --no-resume\
  --use-offline-data \
  --offline-dataset-subset-num=50 \
  --libero-data-dir=/data/libero_goal \
  --libero-task-suite=libero_goal \
  --offline-steps=1000 \
  --env-reuse-frequency=1 \
  --rollout-interval=10 \
  --eval-interval=500 \
  --eval-episodes=5 \
  --max-timesteps=300 \
  --num-steps-wait=10 \
  --capacity=100_000 \
  --lr-schedule-actor.warmup-steps=500 \
  --lr-schedule-actor.peak-lr=2.5e-5 \
  --lr-schedule-actor.decay-steps=30_000 \
  --lr-schedule-actor.decay-lr=2.5e-6 \
  --lr-schedule-critic.warmup-steps=500 \
  --lr-schedule-critic.peak-lr=1.0e-4 \
  --lr-schedule-critic.decay-steps=30_000 \
  --lr-schedule-critic.decay-lr=1.0e-5 \
  --lr-schedule-edit-actor.warmup-steps=500 \
  --lr-schedule-edit-actor.peak-lr=1.0e-5 \
  --lr-schedule-edit-actor.decay-steps=30_000 \
  --lr-schedule-edit-actor.decay-lr=1.0e-6 \
  --lr-schedule-temp.warmup-steps=500 \
  --lr-schedule-temp.peak-lr=1.0e-6 \
  --lr-schedule-temp.decay-steps=30_000 \
  --lr-schedule-temp.decay-lr=1.0e-7 \
  \
  --buffer.capacity-total=50_000 \
  --buffer.batch-offline-ratio=0.5 \
  --buffer.success-memory-per-task=10 \
  --buffer.eviction=fifo \
  \
  --retrieval.enabled \
  --retrieval.topk=8 \
  --retrieval.success-only \
  --retrieval.alpha-q=0.3 \
  --retrieval.refresh-every=2000 \
  --retrieval.use-for-targets \
  --retrieval.use-for-rollout \