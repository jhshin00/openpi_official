#!/usr/bin/env python3
"""
UR3 Robot Control Script using Gello Teleoperator + OpenPI Client
훈련된 UR3 policy를 실제 로봇에 적용하는 script
gello 환경을 사용하여 실시간 카메라 입력과 로봇 상태를 받아서 policy inference 후 로봇 제어
"""

import datetime
import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import threading

import numpy as np
import tyro
import h5py
import cv2
import pyrealsense2 as rs

# Gello 관련 라이브러리들
try:
    from gello.agents.agent import DummyAgent
    from gello.agents.gello_agent import GelloAgent
    from gello.env import RobotEnv
    from gello.robots.robot import PrintRobot
    from gello.zmq_core.robot_node import ZMQClientRobot
    from gello.zmq_core.camera_node import ZMQClientCamera, ZMQServerCamera
    from gello.cameras.camera import CameraDriver
    GELLO_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: Gello libraries not found. Install them for robot control.")
    GELLO_AVAILABLE = False

# OpenPI Client
try:
    from openpi_client import websocket_client_policy as _websocket_client_policy
    OPENPI_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: OpenPI client not found. Install it with: pip install openpi_client")
    OPENPI_AVAILABLE = False

def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    # Policy server configuration
    policy_host: str = "localhost"
    policy_port: int = 8000
    api_key: Optional[str] = None
    
    # Task configuration
    task_prompt: str = "pick up the green grape and put it in the gray pot"
    
    # Robot configuration
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    hz: int = 40  # UR3 제어 주파수 (더 높은 주파수로 부드러운 움직임)
    
    # Agent configuration
    agent: str = "policy"  # "none", "gello", "policy"
    gello_port: Optional[str] = None
    start_joints: Optional[Tuple[float, ...]] = None
    
    # Camera configuration
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    
    # Control parameters
    max_joint_delta: float = 0.8
    open_loop_horizon: int = 5  # 몇 개의 action을 실행한 후 새로운 chunk 요청할지
    smoothing_factor: float = 0.3  # Action smoothing factor (0.1~0.5, 낮을수록 부드러움)
    max_action_velocity: float = 0.1  # 최대 action velocity 제한
    interpolation_steps: int = 3  # Action 간 보간 단계 수 (높을수록 부드러움)
    
    # Data saving
    use_save_interface: bool = True
    data_dir: str = "~/ur3_data"
    
    # Mock mode
    mock: bool = False


class RealSenseDriver(CameraDriver):
    def __init__(self, serial: str, width=640, height=480, fps=30):
        self.serial = serial
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(cfg)

    def read(self, img_size=None):
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        img = np.asanyarray(color.get_data())
        if img_size is not None:
            img = cv2.resize(img, (img_size[1], img_size[0]))
        return img

    def __str__(self):
        return f"RealSense({self.serial})"

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def __del__(self):
        self.close()


def start_server(port, driver):
    server = ZMQServerCamera(driver, port=port)
    server.serve()


class AsyncCamera:
    def __init__(self, client: ZMQClientCamera, target_size=(224, 224), crop_center=None):
        self.client = client
        self.target_size = target_size
        h, w = 320, 320
        self.frame = None
        self.lock = threading.Lock()

        self.crop_center = crop_center

        if crop_center is not None:
            cy, cx = crop_center
            self.x1 = max(cx - w // 2, 0)
            self.x2 = min(cx + w // 2, 640)
            self.y1 = max(cy - h // 2, 0)
            self.y2 = min(cy + h // 2, 480)
        else:
            self.x1 = self.x2 = self.y1 = self.y2 = None

        # 데몬 쓰레드로 백그라운드에서 프레임 갱신
        t = threading.Thread(target=self._update_loop, daemon=True)
        t.start()

    def _update_loop(self):
        while True:
            try:
                raw = self.client.read()
                if raw is None:
                    time.sleep(0.002)
                    continue
                if self.crop_center is not None:
                    crop = raw[self.y1:self.y2, self.x1:self.x2]
                    img = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
                else:
                    img = cv2.resize(raw, self.target_size, interpolation=cv2.INTER_LINEAR)

                with self.lock:
                    self.frame = img
            except Exception as e:
                time.sleep(0.01)

    def read(self):
        # 최초 프레임 없으면 잠깐 대기
        if self.frame is None:
            for _ in range(50):
                time.sleep(0.005)
                if self.frame is not None:
                    break
        with self.lock:
            return None if self.frame is None else self.frame.copy()


class PolicyAgent:
    """OpenPI Policy를 사용하는 에이전트 - Smooth Action Interpolation"""
    
    def __init__(self, host: str, port: int, api_key: Optional[str] = None, task_prompt: str = "pick up the green grape and put it in the gray pot", open_loop_horizon: int = 5):
        if not OPENPI_AVAILABLE:
            raise ImportError("OpenPI client not available")
        
        self.policy = _websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port,
            api_key=api_key,
        )
        
        # 🎯 Task prompt 검증 및 정리
        if not task_prompt or len(task_prompt.strip()) == 0:
            raise ValueError("Task prompt cannot be empty!")
        
        self.task_prompt = task_prompt.strip()
        self.open_loop_horizon = open_loop_horizon
        
        # Simple action buffer
        self.action_buffer = None
        self.action_idx = 0
        
        # Action Smoothing을 위한 상태
        self.last_smoothed_action = None
        self.smoothing_factor = 0.3  # 낮을수록 부드러움 (0.1~0.5)
        self.max_velocity = 0.1  # 최대 joint velocity 제한
        
        # 🎯 Action Interpolation을 위한 상태
        self.interpolation_steps = 3  # action 간 보간 단계 수
        self.interpolation_counter = 0  # 현재 보간 단계
        self.current_action = None  # 현재 실행 중인 action
        self.next_action = None  # 다음 action
        
        print(f"✅ Policy server connected to {host}:{port}")
        print(f"📝 Task: '{self.task_prompt}'")
        print(f"🔄 Short horizon: {self.open_loop_horizon} actions")
        print(f"🚀 Smooth Action Interpolation + Smoothing mode enabled")
        print(f"🔄 Smoothing factor: {self.smoothing_factor}")
        print(f"🔄 Max velocity: {self.max_velocity}")
        print(f"🔄 Interpolation steps: {self.interpolation_steps}")
        
        # 서버 메타데이터 확인
        try:
            metadata = self.policy.get_server_metadata()
            print(f"📋 Server metadata: {metadata}")
            
            # 🚨 Policy 서버가 task prompt를 지원하는지 확인
            if hasattr(metadata, 'get') and metadata.get('supports_task_prompt', False):
                print("✅ Policy server supports task prompts")
            else:
                print("⚠️  Policy server may not support task prompts")
                
        except Exception as e:
            print(f"⚠️  Could not get server metadata: {e}")
    
    def act(self, obs: dict) -> np.ndarray:
        """Smooth Action Interpolation 방식으로 부드러운 action 제공"""
        try:
            current_joints = obs["joint_positions"]
            
            # Action buffer가 비어있거나 완료되면 새로운 action 요청
            if self.action_buffer is None or self.action_idx >= len(self.action_buffer):
                print(f"\n🔄 Requesting new actions (horizon: {self.open_loop_horizon})")
                self._request_new_actions(obs)
                
                if self.action_buffer is None:
                    return np.array(current_joints, dtype=np.float32)
                
                # 새로운 action buffer 시작 시 초기화
                self.interpolation_counter = 0
                self.current_action = current_joints.copy()
                self.next_action = np.array(self.action_buffer[0], dtype=np.float32)
            
            # 🎯 Action Interpolation 적용
            if self.interpolation_counter < self.interpolation_steps:
                # 보간 단계 실행
                interpolated_action = self._interpolate_action(
                    self.current_action, 
                    self.next_action, 
                    self.interpolation_counter, 
                    self.interpolation_steps
                )
                
                self.interpolation_counter += 1
                
                # Action Smoothing 적용
                smoothed_action = self._smooth_action(interpolated_action, current_joints)
                
                # Velocity 제한 적용
                final_action = self._limit_velocity(smoothed_action, current_joints)
                
                print(f"  🎯 Interpolation step {self.interpolation_counter}/{self.interpolation_steps}: {final_action[:3]}...")
                return final_action
            
            else:
                # 보간 완료, 다음 action으로 진행
                self.action_idx += 1
                self.interpolation_counter = 0
                
                if self.action_idx < len(self.action_buffer):
                    # 다음 action으로 보간 시작
                    self.current_action = self.next_action.copy()
                    self.next_action = np.array(self.action_buffer[self.action_idx], dtype=np.float32)
                    
                    # 첫 번째 보간 단계 실행
                    interpolated_action = self._interpolate_action(
                        self.current_action, 
                        self.next_action, 
                        0, 
                        self.interpolation_steps
                    )
                    
                    self.interpolation_counter = 1
                    
                    # Action Smoothing 적용
                    smoothed_action = self._smooth_action(interpolated_action, current_joints)
                    
                    # Velocity 제한 적용
                    final_action = self._limit_velocity(smoothed_action, current_joints)
                    
                    print(f"  🎯 New action {self.action_idx}/{len(self.action_buffer)}: {final_action[:3]}...")
                    return final_action
                
                else:
                    # Action buffer 완료
                    return np.array(current_joints, dtype=np.float32)
            
        except Exception as e:
            print(f"❌ Policy inference failed: {e}")
            return np.array(obs["joint_positions"], dtype=np.float32)
    
    def _interpolate_action(self, start_action: np.ndarray, end_action: np.ndarray, step: int, total_steps: int) -> np.ndarray:
        """두 action 간 부드러운 보간"""
        if total_steps <= 1:
            return end_action
        
        # Linear interpolation with easing
        alpha = step / (total_steps - 1)
        
        # Easing function for smoother transition
        eased_alpha = self._ease_in_out(alpha)
        
        # 보간된 action 계산
        interpolated = start_action + eased_alpha * (end_action - start_action)
        
        return interpolated
    
    def _ease_in_out(self, t: float) -> float:
        """부드러운 easing 함수"""
        # Smooth step function
        return t * t * (3.0 - 2.0 * t)
    
    def _smooth_action(self, target_action: np.ndarray, current_joints: np.ndarray) -> np.ndarray:
        """Action을 부드럽게 평활화"""
        if self.last_smoothed_action is None:
            # 첫 번째 실행 시
            self.last_smoothed_action = current_joints.copy()
        
        # Exponential smoothing 적용
        # smoothed = α * target + (1-α) * last_smoothed
        smoothed = (self.smoothing_factor * target_action + 
                   (1 - self.smoothing_factor) * self.last_smoothed_action)
        
        # 결과 저장
        self.last_smoothed_action = smoothed.copy()
        
        return smoothed
    
    def _limit_velocity(self, target_action: np.ndarray, current_joints: np.ndarray) -> np.ndarray:
        """Joint velocity를 제한하여 급격한 변화 방지"""
        # 현재 joint에서 target까지의 변화량
        delta = target_action - current_joints
        
        # 최대 velocity 제한
        max_delta = self.max_velocity
        delta_magnitude = np.linalg.norm(delta)
        
        if delta_magnitude > max_delta:
            # Velocity 제한 적용
            delta = delta / delta_magnitude * max_delta
            target_action = current_joints + delta
        
        return target_action
    
    def _request_new_actions(self, obs: dict):
        """현재 observation으로 새로운 actions 요청"""
        try:
            print(f"  🔄 Requesting new actions with current state...")
            
            # Policy 입력 형식으로 변환
            formatted_obs = self._format_observation_for_policy(obs)
            
            # DEBUG: Observation 데이터 확인
            print(f"  📷 Base image shape: {formatted_obs['observation/base_image'].shape if formatted_obs['observation/base_image'] is not None else 'None'}")
            print(f"  📷 Wrist image shape: {formatted_obs['observation/wrist_image'].shape if formatted_obs['observation/wrist_image'] is not None else 'None'}")
            print(f"  🤖 Joint positions: {formatted_obs['observation/state']}")
            
            # 🎯 TASK PROMPT 상세 확인
            print(f"\n🎯 TASK PROMPT VERIFICATION:")
            print(f"  📝 Original task prompt: '{self.task_prompt}'")
            print(f"  📝 Formatted task prompt: '{formatted_obs['prompt']}'")
            print(f"  📝 Task prompt length: {len(formatted_obs['prompt'])} characters")
            print(f"  📝 Task prompt type: {type(formatted_obs['prompt'])}")
            
            # Policy inference
            result = self.policy.infer(formatted_obs)
            
            # DEBUG: Policy 응답 상세 분석
            print(f"  🎯 Policy response analysis:")
            print(f"    📋 Response keys: {list(result.keys())}")
            
            if "actions" in result:
                actions = result["actions"]
                print(f"    🎯 Actions shape: {actions.shape if hasattr(actions, 'shape') else len(actions)}")
                print(f"    🎯 First action: {actions[0] if len(actions) > 0 else 'None'}")
                
                # 🚨 Task prompt 반영 여부 확인
                if "task_info" in result:
                    print(f"    📝 Task info from policy: {result['task_info']}")
                else:
                    print(f"    ⚠️  No task_info in policy response")
                
                if "prompt_used" in result:
                    print(f"    📝 Prompt used by policy: {result['prompt_used']}")
                else:
                    print(f"    ⚠️  No prompt_used in policy response")
                
                # Action buffer 설정
                self.action_buffer = actions
                self.action_idx = 0
                print(f"    ✅ New actions loaded: {len(self.action_buffer)} actions")
            else:
                print("    ❌ No actions received from policy")
                print(f"    📋 Full response: {result}")
                
        except Exception as e:
            print(f"  ❌ Failed to request new actions: {e}")
            import traceback
            traceback.print_exc()
    
    def _format_observation_for_policy(self, obs: dict) -> dict:
        """Policy가 기대하는 형식으로 observation을 변환합니다"""
        # BGR → RGB 변환
        base_image = obs.get("base_rgb")
        wrist_image = obs.get("wrist_rgb")
        
        if base_image is not None:
            base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
        if wrist_image is not None:
            wrist_image = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)
        
        # 🎯 UR3 Policy 전용 형식으로 변경
        # 참고: src/openpi/training/config.py의 LeRobotUR3DataConfig와 일치
        formatted_obs = {
            "observation/base_image": base_image,      # base camera
            "observation/wrist_image": wrist_image,    # wrist camera
            "observation/state": obs["joint_positions"], # joint positions
            "prompt": self.task_prompt  # task prompt
        }
        
        # 🚨 Task prompt 검증
        if not self.task_prompt or len(self.task_prompt.strip()) == 0:
            print("⚠️  WARNING: Task prompt is empty or None!")
            print(f"  Task prompt: '{self.task_prompt}'")
            print(f"  Task prompt type: {type(self.task_prompt)}")
            print(f"  Task prompt length: {len(self.task_prompt) if self.task_prompt else 0}")
        
        # 🎯 Observation 데이터 검증
        print(f"  🔍 Observation format verification (UR3):")
        print(f"    📷 Base image: {formatted_obs['observation/base_image'].shape if formatted_obs['observation/base_image'] is not None else 'None'}")
        print(f"    📷 Wrist image: {formatted_obs['observation/wrist_image'].shape if formatted_obs['observation/wrist_image'] is not None else 'None'}")
        print(f"    🤖 Joint positions: {formatted_obs['observation/state']}")
        print(f"    📝 Task prompt: '{formatted_obs['prompt']}'")
        
        return formatted_obs


def main(args):
    if not GELLO_AVAILABLE:
        print("❌ Gello libraries are required")
        return
    
    if args.agent == "policy" and not OPENPI_AVAILABLE:
        print("❌ OpenPI client is required for policy agent")
        return
    
    # 로봇과 카메라 초기화
    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        # RealSense 카메라 초기화
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) < 2:
            print("❌ 두 대 이상의 RealSense가 연결되어 있지 않습니다.")
            return
        
        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devs[:2]]
        drivers = [RealSenseDriver(s, args.camera_width, args.camera_height, args.camera_fps) 
                  for s in serials]
        
        # 카메라 서버 시작
        ports = [args.wrist_camera_port, args.base_camera_port]
        threads = []
        for port, drv in zip(ports, drivers):
            t = threading.Thread(target=start_server, args=(port, drv), daemon=True)
            t.start()
            threads.append(t)

        time.sleep(1)  # 바인딩 대기

        # 카메라 클라이언트 연결
        client1 = ZMQClientCamera(port=args.wrist_camera_port)
        client2 = ZMQClientCamera(port=args.base_camera_port)
        
        camera_clients = {
            "wrist": AsyncCamera(client1, crop_center=(320, 240)),
            "base": AsyncCamera(client2),
        }
        
        # DEBUG: 카메라 연결 상태 확인
        print("\n📷 DEBUG: Camera Status")
        print(f"  📷 Wrist camera port: {args.wrist_camera_port}")
        print(f"  📷 Base camera port: {args.base_camera_port}")
        print(f"  📷 Camera clients: {list(camera_clients.keys())}")
        
        # 카메라 프레임 테스트
        time.sleep(2)  # 카메라 초기화 대기
        for cam_name, cam in camera_clients.items():
            frame = cam.read()
            if frame is not None:
                print(f"  ✅ {cam_name} camera: frame shape {frame.shape}, dtype {frame.dtype}")
            else:
                print(f"  ❌ {cam_name} camera: no frame received")
        
        # 로봇 클라이언트 연결
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    
    # Gello 환경 생성
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)
    
    # 에이전트 초기화
    if args.agent == "policy":
        print(f"\n🎯 POLICY AGENT INITIALIZATION:")
        print(f"  📝 Task prompt: '{args.task_prompt}'")
        print(f"  📝 Task prompt length: {len(args.task_prompt)} characters")
        print(f"  📝 Task prompt type: {type(args.task_prompt)}")
        print(f"  📝 Policy host: {args.policy_host}:{args.policy_port}")
        
        # 🚨 Task prompt 검증
        if not args.task_prompt or len(args.task_prompt.strip()) == 0:
            print("  ❌ ERROR: Task prompt is empty or None!")
            print("  ❌ This will cause the policy to ignore task instructions!")
            return
        elif len(args.task_prompt.strip()) < 10:
            print("  ⚠️  WARNING: Task prompt seems too short!")
            print("  ⚠️  Consider using a more descriptive task description")
        
        # Task prompt가 기본값인지 확인
        default_prompt = "pick up the green grape and put it in the gray pot"
        if args.task_prompt == default_prompt:
            print("  ⚠️  WARNING: Using default task prompt!")
            print("  ⚠️  Consider specifying a custom task with --task-prompt")
        
        agent = PolicyAgent(args.policy_host, args.policy_port, args.api_key, args.task_prompt, args.open_loop_horizon)
        # Smoothing 파라미터 설정
        agent.smoothing_factor = args.smoothing_factor
        agent.max_velocity = args.max_action_velocity
        # Interpolation 파라미터 설정
        agent.interpolation_steps = args.interpolation_steps
    elif args.agent == "gello":
        gello_port = args.gello_port
        if gello_port is None:
            usb_ports = glob.glob("/dev/serial/by-id/*")
            print(f"Found {len(usb_ports)} ports")
            if len(usb_ports) > 0:
                gello_port = usb_ports[0]
                print(f"using port {gello_port}")
            else:
                raise ValueError("No gello port found, please specify one or plug in gello")
        
        if args.start_joints is None:
            reset_joints = np.deg2rad([0, -90, -90, -90, 90, 180])
        else:
            reset_joints = args.start_joints
        
        agent = GelloAgent(port=gello_port, start_joints=args.start_joints)
        
        # 초기 위치로 이동
        curr_joints = env.get_obs()["joint_positions"]
        if reset_joints.shape == curr_joints.shape:
            max_delta = (np.abs(curr_joints - reset_joints)).max()
            steps = min(int(max_delta / 0.01), 100)
            for jnt in np.linspace(curr_joints, reset_joints, steps):
                env.step(jnt)
                time.sleep(0.001)
    
    elif args.agent == "dummy" or args.agent == "none":
        agent = DummyAgent(num_dofs=robot_client.num_dofs())
    else:
        raise ValueError(f"Invalid agent name: {args.agent}")
    
    # 시작 위치로 이동
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    if abs_deltas[id_max_joint_delta] > args.max_joint_delta:
        id_mask = abs_deltas > args.max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}")
        return

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(joints), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    # 부드러운 시작을 위한 점진적 이동
    max_delta = 0.05
    for _ in range(25):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    # 최종 위치 확인
    obs = env.get_obs()
    joints = obs["joint_positions"]
    action = agent.act(obs)
    if (action - joints > 0.5).any():
        print("Action is too big")
        joint_index = np.where(action - joints > 0.8)
        for j in joint_index:
            print(f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}")
        exit()

    # 키보드 인터페이스 초기화
    if args.use_save_interface:
        from gello.data_utils.keyboard_interface import KBReset
        kb_interface = KBReset()

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    # 메인 제어 루프
    save_path = None
    buffer: List[Tuple[Dict[str, Any], np.ndarray]] = []
    recording: bool = False
    start_time = time.time()
    
    while True:
        # 시간 표시
        num = time.time() - start_time
        message = f"\rTime passed: {round(num, 2)}          "
        print_color(message, color="white", attrs=("bold",), end="", flush=True)
        
        # 에이전트로부터 action 가져오기
        action = agent.act(obs)
        
        # DEBUG: 주기적 observation 상태 확인 (10초마다)
        if int(time.time() - start_time) % 10 == 0:
            print(f"\n🔍 DEBUG: Current Observation Status (t={int(time.time() - start_time)}s)")
            print(f"  📷 Base RGB available: {'base_rgb' in obs}")
            print(f"  📷 Wrist RGB available: {'wrist_rgb' in obs}")
            print(f"  🤖 Joint positions available: {'joint_positions' in obs}")
            if 'joint_positions' in obs:
                print(f"  🤖 Current joints: {obs['joint_positions']}")
            if 'base_rgb' in obs and obs['base_rgb'] is not None:
                print(f"  📷 Base image: {obs['base_rgb'].shape}, dtype: {obs['base_rgb'].dtype}")
            if 'wrist_rgb' in obs and obs['wrist_rgb'] is not None:
                print(f"  📷 Wrist image: {obs['wrist_rgb'].shape}, dtype: {obs['wrist_rgb'].dtype}")
        
        # 데이터 저장 인터페이스
        if args.use_save_interface:
            state = kb_interface.update()
            if state == "start":
                dt_time = datetime.datetime.now()
                save_path = (
                    Path(args.data_dir).expanduser()
                    / args.agent
                    / dt_time.strftime("%m%d_%H%M%S")
                )
                save_path.mkdir(parents=True, exist_ok=True)
                print(f"Recording to {save_path}")
                buffer.clear()
                recording = True

            elif state == "save":
                assert save_path is not None, "something went wrong"
                buffer.append((obs.copy(), action.copy()))  

            elif state == "normal":
                if recording:
                    # End of recording: flush buffer to HDF5
                    if save_path and buffer:
                        # Create HDF5 file
                        h5_file = save_path / "data.hdf5"
                        with h5py.File(h5_file, 'w') as f:
                            grp = f.create_group('data')
                            # keys from first obs
                            keys = list(buffer[0][0].keys())
                            # prepare arrays
                            for key in keys:
                                # stack obs values
                                data_arr = np.stack([b[0][key] for b in buffer], axis=0)
                                grp.create_dataset(key, data=data_arr)
                            # actions
                            acts = np.stack([b[1] for b in buffer], axis=0)
                            grp.create_dataset('actions', data=acts)
                        
                        # 동일 폴더에 RGB 비디오로도 저장
                        fps = 30
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                        # base 와 wrist 각 카메라별로 비디오 생성
                        for cam in ('base', 'wrist'):
                            frames = [b[0][f'{cam}_rgb'] for b in buffer]
                            h, w = frames[0].shape[:2]
                            video_path = save_path / f"{cam}_rgb.mp4"
                            vw = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                            for frame in frames:
                                vw.write(frame)
                            vw.release()
                    
                    # reset
                    recording = False
                    buffer.clear()
                    save_path = None
            else:
                raise ValueError(f"Invalid state {state}")
        
        # Action 실행
        print("start")
        t0 = time.perf_counter()
        obs = env.step(action)
        
        # 고정 주기 제어 (이전 방식으로 복원)
        dt_actual = time.perf_counter() - t0
        dt_target = 1.0 / args.hz
        
        # 제어 주파수 맞추기
        sleep_time = dt_target - dt_actual
        if sleep_time > 0:
            time.sleep(sleep_time)
            total_time = time.perf_counter() - t0
            print(f"[Timing] env.step(): {dt_actual*1000:.1f}ms + sleep: {sleep_time*1000:.1f}ms = 총 {total_time*1000:.1f}ms (목표: {dt_target*1000:.1f}ms)")
        else:
            print(f"[Warning] ⚠️  처리 시간이 {dt_actual*1000:.1f}ms로 목표 주기 {dt_target*1000:.1f}ms를 초과했습니다.")

        print("end")


if __name__ == "__main__":
    main(tyro.cli(Args))