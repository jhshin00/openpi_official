#!/bin/bash
set -euo pipefail

# UR3 Robot Control Script
# 실제 로봇에서 policy를 실행하기 위한 script

export CUDA_VISIBLE_DEVICES=2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_LEROBOT_HOME=./datasets
export HF_DATASETS_CACHE=$(pwd)/datasets/cache
export HUGGINGFACE_HUB_CACHE=$(pwd)/datasets/cache/hub

echo "🤖 UR3 Robot Control Script"
echo "=========================="
echo "⚠️  WARNING: This will control a real robot!"
echo "⚠️  Make sure the robot is in a safe position and area is clear"
echo "⚠️  Press Ctrl+C to stop the robot safely"
echo "=========================="

# 사용자 확인
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Robot control cancelled"
    exit 1
fi

# 추가 안전 확인
read -p "Is the robot workspace clear of obstacles and people? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Safety check failed - robot control cancelled"
    exit 1
fi

echo "✅ Safety checks passed - starting robot control..."

# 가상환경 활성화 및 로봇 제어 실행
source .venv/bin/activate

# 로봇 제어 실행
python run_ur3_robot.py \
  --host localhost \
  --port 8000 \
  --robot_ip "192.168.1.100" \
  --control_freq 10.0 \
  --base_camera_id 0 \
  --wrist_camera_id 1 \
  --task_prompt "pick up the green grape and put it in the gray pot" \
  --max_joint_velocity 0.5 \
  --max_joint_acceleration 1.0 \
  --action_horizon 8 \
  --safety_check true 