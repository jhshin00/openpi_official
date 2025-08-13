set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

./.venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config   pi0_fql_libero_lora_finetune \
  --policy.dir      checkpoints/pi0_fql_libero_lora_finetune/exp_rejection_chunk_0728_1839/29999