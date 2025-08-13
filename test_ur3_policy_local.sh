set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_LEROBOT_HOME=./datasets
export HF_DATASETS_CACHE=$(pwd)/datasets/cache
export HUGGINGFACE_HUB_CACHE=$(pwd)/datasets//cache/hub

echo "UR3 Policy Test in local (Main)"

./.venv/bin/python test_ur3_policy.py \
  --test_mode   REAL_DATA \
  --data_dir    ./datasets/ur3_datasets \
  --task_name   TASK_3_pick_up_green_grape_and_put_it_in_the_gray_pot \
  --num_steps   5