#!/usr/bin/env python3
"""
UR3 Robot Control Script
훈련된 UR3 policy를 실제 로봇에 적용하는 script
실시간 카메라 입력과 로봇 상태를 받아서 policy inference 후 로봇 제어
"""

import dataclasses
import logging
import pathlib
import time
from typing import Optional

import numpy as np
import tyro
from openpi_client import websocket_client_policy as _websocket_client_policy

# UR3 로봇 제어를 위한 라이브러리들
try:
    import cv2
    from ur_control import UR3Controller  # UR3 제어 라이브러리 (가정)
except ImportError:
    print("⚠️  Warning: UR3 control libraries not found. Install them for real robot control.")
    cv2 = None
    UR3Controller = None

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RobotArgs:
    """Command line arguments for robot control."""
    
    # Policy server configuration
    host: str = "localhost"
    port: int = 8000
    api_key: Optional[str] = None
    
    # Robot configuration
    robot_ip: str = "192.168.1.100"  # UR3 로봇 IP 주소
    control_freq: float = 10.0  # 제어 주파수 (Hz)
    
    # Camera configuration
    base_camera_id: int = 0  # 베이스 카메라 ID
    wrist_camera_id: int = 1  # 손목 카메라 ID
    
    # Task configuration
    task_prompt: str = "pick up the green grape and put it in the gray pot"
    
    # Safety configuration
    max_joint_velocity: float = 0.5  # 최대 관절 속도 (rad/s)
    max_joint_acceleration: float = 1.0  # 최대 관절 가속도 (rad/s²)
    
    # Control parameters
    action_horizon: int = 8  # 한 번에 실행할 action 개수
    safety_check: bool = True  # 안전 검사 활성화


class UR3RobotController:
    """UR3 로봇을 제어하는 클래스."""
    
    def __init__(self, args: RobotArgs):
        self.args = args
        self.control_dt = 1.0 / args.control_freq
        
        # Policy 서버 연결
        self.policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
        )
        
        # 로봇 제어기 초기화
        if UR3Controller:
            self.robot = UR3Controller(
                robot_ip=args.robot_ip,
                max_velocity=args.max_joint_velocity,
                max_acceleration=args.max_joint_acceleration
            )
            self.robot_connected = True
        else:
            self.robot = None
            self.robot_connected = False
            logger.warning("Robot controller not available - running in simulation mode")
        
        # 카메라 초기화
        self.base_camera = None
        self.wrist_camera = None
        self._init_cameras()
        
        # 제어 상태
        self.is_running = False
        self.current_action_idx = 0
        self.action_buffer = None
        
    def _init_cameras(self):
        """카메라를 초기화합니다."""
        if cv2 is None:
            logger.warning("OpenCV not available - using dummy camera data")
            return
            
        try:
            # 베이스 카메라
            self.base_camera = cv2.VideoCapture(self.args.base_camera_id)
            if not self.base_camera.isOpened():
                logger.warning(f"Failed to open base camera {self.args.base_camera_id}")
                self.base_camera = None
            
            # 손목 카메라
            self.wrist_camera = cv2.VideoCapture(self.args.wrist_camera_id)
            if not self.wrist_camera.isOpened():
                logger.warning(f"Failed to open wrist camera {self.args.wrist_camera_id}")
                self.wrist_camera = None
                
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
    
    def get_observation(self) -> dict:
        """현재 로봇 상태와 카메라 이미지를 가져옵니다."""
        obs = {}
        
        # 카메라 이미지
        if self.base_camera and self.base_camera.isOpened():
            ret, frame = self.base_camera.read()
            if ret:
                # BGR to RGB 변환 및 리사이즈
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                obs["observation/base_image"] = frame_resized
            else:
                obs["observation/base_image"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
        else:
            obs["observation/base_image"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
        
        if self.wrist_camera and self.wrist_camera.isOpened():
            ret, frame = self.wrist_camera.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                obs["observation/wrist_image"] = frame_resized
            else:
                obs["observation/wrist_image"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
        else:
            obs["observation/wrist_image"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
        
        # 로봇 상태 (joint positions)
        if self.robot_connected:
            try:
                joint_positions = self.robot.get_joint_positions()
                obs["observation/state"] = joint_positions[:7]  # 7개 관절만 사용
            except Exception as e:
                logger.warning(f"Failed to get robot state: {e}")
                obs["observation/state"] = np.random.rand(7).astype(np.float32)
        else:
            obs["observation/state"] = np.random.rand(7).astype(np.float32)
        
        # Task prompt
        obs["prompt"] = self.args.task_prompt
        
        return obs
    
    def execute_action(self, action: np.ndarray) -> bool:
        """Action을 로봇에 실행합니다."""
        if not self.robot_connected:
            logger.info(f"Simulation mode - would execute action: {action[:3]}...")
            return True
        
        try:
            # Action을 7차원 joint positions로 변환
            joint_targets = action[:7]  # 7개 관절 위치
            
            # 안전 검사
            if self.args.safety_check:
                if not self._safety_check(joint_targets):
                    logger.warning("Safety check failed - action rejected")
                    return False
            
            # 로봇 제어
            self.robot.move_joints(joint_targets, blocking=False)
            return True
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False
    
    def _safety_check(self, joint_targets: np.ndarray) -> bool:
        """Action의 안전성을 검사합니다."""
        try:
            current_joints = self.robot.get_joint_positions()
            
            # Joint limit 검사
            joint_limits = self.robot.get_joint_limits()
            for i, (target, current, limits) in enumerate(zip(joint_targets, current_joints, joint_limits)):
                if not (limits[0] <= target <= limits[1]):
                    logger.warning(f"Joint {i} target {target:.3f} exceeds limits {limits}")
                    return False
            
            # Sudden movement 검사
            joint_diff = np.abs(joint_targets - current_joints)
            max_allowed_diff = self.args.max_joint_velocity * self.control_dt
            
            if np.any(joint_diff > max_allowed_diff):
                logger.warning(f"Joint movement too large: {joint_diff} > {max_allowed_diff}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return False
    
    def run_control_loop(self):
        """메인 제어 루프를 실행합니다."""
        print("🤖 UR3 Robot Control Started!")
        print("=" * 60)
        print(f"🔌 Policy server: {self.args.host}:{self.args.port}")
        print(f"🤖 Robot IP: {self.args.robot_ip}")
        print(f"📷 Base camera: {self.args.base_camera_id}, Wrist camera: {self.args.wrist_camera_id}")
        print(f"⚡ Control frequency: {self.args.control_freq} Hz")
        print(f"📝 Task: {self.args.task_prompt}")
        print("=" * 60)
        
        if not self.robot_connected:
            print("⚠️  Running in simulation mode (no real robot)")
        
        self.is_running = True
        last_policy_call = 0
        
        try:
            while self.is_running:
                loop_start = time.time()
                
                # Policy에서 새로운 action 가져오기
                if (self.action_buffer is None or 
                    self.current_action_idx >= len(self.action_buffer) or
                    time.time() - last_policy_call > 1.0):  # 1초마다 policy 호출
                    
                    # Observation 가져오기
                    obs = self.get_observation()
                    
                    # Policy inference
                    try:
                        result = self.policy.infer(obs)
                        actions = result["actions"]  # Shape: (50, 7)
                        
                        # Action buffer 업데이트
                        self.action_buffer = actions
                        self.current_action_idx = 0
                        last_policy_call = time.time()
                        
                        print(f"🔄 New policy action: {actions.shape}, first action: {actions[0, :3]}")
                        
                    except Exception as e:
                        logger.error(f"Policy inference failed: {e}")
                        time.sleep(0.1)
                        continue
                
                # Action 실행
                if self.action_buffer is not None:
                    current_action = self.action_buffer[self.current_action_idx]
                    
                    if self.execute_action(current_action):
                        self.current_action_idx += 1
                        print(f"✅ Action {self.current_action_idx}/{len(self.action_buffer)} executed")
                    else:
                        print("❌ Action execution failed")
                
                # 제어 주파수 맞추기
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.control_dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Ctrl+C로 종료
                if not self.is_running:
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Control loop interrupted by user")
        except Exception as e:
            logger.error(f"Control loop error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스를 정리합니다."""
        print("🧹 Cleaning up...")
        
        if self.robot_connected and self.robot:
            try:
                # 로봇을 안전한 위치로 이동
                safe_position = [0, -np.pi/2, 0, -np.pi/2, 0, 0, 0]
                self.robot.move_joints(safe_position, blocking=True)
                print("✅ Robot moved to safe position")
            except Exception as e:
                logger.error(f"Failed to move robot to safe position: {e}")
        
        if self.base_camera:
            self.base_camera.release()
        if self.wrist_camera:
            self.wrist_camera.release()
        
        print("✅ Cleanup completed")


def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO)
    
    args = tyro.cli(RobotArgs)
    
    # Policy 서버 연결 테스트
    try:
        controller = UR3RobotController(args)
        print("✅ Robot controller initialized successfully")
        
        # 제어 루프 실행
        controller.run_control_loop()
        
    except Exception as e:
        print(f"❌ Failed to initialize robot controller: {e}")
        logger.error(f"Initialization failed: {e}", exc_info=True)


if __name__ == "__main__":
    main() 