set -euo pipefail

export CUDA_VISIBLE_DEVICES=1,2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_LEROBOT_HOME=./datasets
export HF_DATASETS_CACHE=$(pwd)/datasets/cache
export HUGGINGFACE_HUB_CACHE=$(pwd)/datasets//cache/hub

./.venv/bin/python test_data.py 
# /ssd1/openpi_official/datasets/libero_fql/data/chunk-000/episode_000019.parquet \
#     --head 100



# /ssd1/openpi_official/datasets/libero_fql/data/chunk-000/episode_000139.parquet \
#     --head 5