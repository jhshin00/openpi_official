set -euo pipefail

# # Setup libero environment
# echo "Setting up libero environment..."
# uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
# uv pip install -e packages/openpi-client
# uv pip install -e third_party/libero
# echo "Libero environment setup complete!"

# export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

export CUDA_VISIBLE_DEVICES=0

examples/libero/.venv/bin/python examples/libero/main.py \
  --args.host "0.0.0.0" \
  --args.port 8000 \
  --args.resize_size            224 \
  --args.replan_steps           5 \
  --args.task_suite_name        libero_10 \
  --args.num_steps_wait         10 \
  --args.num_trials_per_task    10 \
  --args.video_out_path         data/pi0_fql_libero_lora_finetune/videos/exp_rejection_chunk_0728_1839/libero_10_rejection \
  --args.seed                   42


  