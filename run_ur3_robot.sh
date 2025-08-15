#!/bin/bash
set -euo pipefail

# ===== GPU 및 환경 설정 =====
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
export HF_LEROBOT_HOME=./datasets
export HF_DATASETS_CACHE=$(pwd)/datasets/cache
export HUGGINGFACE_HUB_CACHE=$(pwd)/datasets/cache/hub

# ===== Policy Server 설정 =====
export POLICY_HOST=${POLICY_HOST:-localhost}
export POLICY_PORT=${POLICY_PORT:-8000}
export API_KEY=${API_KEY:-}

# ===== Task 설정 =====
export TASK_PROMPT=${TASK_PROMPT:-"pick up the green grape and put it in the gray pot"}

# ===== Robot 설정 =====
export ROBOT_PORT=${ROBOT_PORT:-6001}
export WRIST_CAMERA_PORT=${WRIST_CAMERA_PORT:-5000}
export BASE_CAMERA_PORT=${BASE_CAMERA_PORT:-5001}
export HOSTNAME=${HOSTNAME:-127.0.0.1}
export HZ=${HZ:-40}

# ===== Agent 설정 =====
export AGENT=${AGENT:-policy}
export GELLO_PORT=${GELLO_PORT:-}
export START_JOINTS=${START_JOINTS:-}

# ===== Camera 설정 =====
export CAMERA_WIDTH=${CAMERA_WIDTH:-640}
export CAMERA_HEIGHT=${CAMERA_HEIGHT:-480}
export CAMERA_FPS=${CAMERA_FPS:-30}

# ===== Control 설정 =====
export MAX_JOINT_DELTA=${MAX_JOINT_DELTA:-0.8}
export OPEN_LOOP_HORIZON=${OPEN_LOOP_HORIZON:-5}

# ===== Data Saving 설정 =====
export USE_SAVE_INTERFACE=${USE_SAVE_INTERFACE:-true}
export DATA_DIR=${DATA_DIR:-~/ur3_data}

# ===== Mock Mode 설정 =====
export MOCK=${MOCK:-false}

echo "🚀 UR3 Robot Control Script"
echo "=========================="
echo "Policy Server: ${POLICY_HOST}:${POLICY_PORT}"
echo "Task: ${TASK_PROMPT}"
echo "Agent: ${AGENT}"
echo "Robot Port: ${ROBOT_PORT}"
echo "Camera: ${CAMERA_WIDTH}x${CAMERA_HEIGHT} @ ${CAMERA_FPS}fps"
echo "Control Hz: ${HZ}"
echo "Mock Mode: ${MOCK}"
echo "=========================="

# ===== main.py 실행 =====
echo "Starting main.py..."
python main.py \
    --policy-host "${POLICY_HOST}" \
    --policy-port "${POLICY_PORT}" \
    --api-key "${API_KEY}" \
    --task-prompt "${TASK_PROMPT}" \
    --robot-port "${ROBOT_PORT}" \
    --wrist-camera-port "${WRIST_CAMERA_PORT}" \
    --base-camera-port "${BASE_CAMERA_PORT}" \
    --hostname "${HOSTNAME}" \
    --hz "${HZ}" \
    --agent "${AGENT}" \
    --gello-port "${GELLO_PORT}" \
    --start-joints "${START_JOINTS}" \
    --camera-width "${CAMERA_WIDTH}" \
    --camera-height "${CAMERA_HEIGHT}" \
    --camera-fps "${CAMERA_FPS}" \
    --max-joint-delta "${MAX_JOINT_DELTA}" \
    --open-loop-horizon "${OPEN_LOOP_HORIZON}" \
    --use-save-interface "${USE_SAVE_INTERFACE}" \
    --data-dir "${DATA_DIR}" \
    --mock "${MOCK}"

echo "✅ UR3 Robot Control finished"