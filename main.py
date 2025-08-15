#!/usr/bin/env python3
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

# Gello
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
    print("⚠️  Warning: Gello libraries not found.")
    GELLO_AVAILABLE = False

# OpenPI client
try:
    from openpi_client import websocket_client_policy as _websocket_client_policy
    OPENPI_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: OpenPI client not found. pip install openpi_client")
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
    hz: int = 40  # UR3 제어 주파수
    
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
            except Exception:
                time.sleep(0.01)

    def read(self):
        if self.frame is None:
            for _ in range(50):
                time.sleep(0.005)
                if self.frame is not None:
                    break
        with self.lock:
            return None if self.frame is None else self.frame.copy()


class PolicyAgent:
    """OpenPI Policy + RTC: prefetch, overlap soft-blend, smoothing, per-joint velocity clamp."""

    def __init__(
        self,
        host: str,
        port: int,
        api_key: Optional[str] = None,
        task_prompt: str = "pick up the green grape and put it in the gray pot",
        open_loop_horizon: int = 5,
    ):
        if not OPENPI_AVAILABLE:
            raise ImportError("OpenPI client not available")
        self.policy = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port, api_key=api_key)

        if not task_prompt or len(task_prompt.strip()) == 0:
            raise ValueError("Task prompt cannot be empty!")

        self.task_prompt = task_prompt.strip()
        self.open_loop_horizon = int(open_loop_horizon)

        # RTC state
        self.action_buffer = None  # np.ndarray [H,D]
        self.action_idx = 0
        self.next_chunk_buffer = None
        self.is_generating_next_chunk = False
        self.chunk_generation_thread = None
        self.rtc_prefetch_ratio = 0.7

        # Inpainting window length (for server API that accepts it; client keeps its own frozen subset too)
        self.enable_flow_inpainting = True
        self.flow_inpaint_steps = 5
        self.frozen_action_window = 2

        # Smoothing
        self.last_smoothed_action = None
        self.smoothing_factor = 0.3
        self.max_velocity = 0.1

        # Interpolation
        self.interpolation_steps = 3
        self.interpolation_counter = 0
        self.current_action = None
        self.next_action = None

        # Overlap soft-blend params (client side)
        self._prev_chunk = None
        self._emin = max(1, self.open_loop_horizon)  # 간단 버전: E_min=open_loop_horizon
        self._overlap_tau = 0.25

        # Gripper action processing
        self.gripper_threshold = 0.5  # Gripper open/close threshold
        self.use_hysteresis = True    # Hysteresis 사용 여부
        self._current_gripper_state = 0.0  # 현재 gripper 상태 (0=open, 1=close)

        # Locks
        self._lock = threading.Lock()

        print(f"✅ Policy server connected to {host}:{port}")
        print(f"📝 Task: '{self.task_prompt}'")
        print(f"🔄 Short horizon: {self.open_loop_horizon} actions")
        print(f"🚀 RTC + smoothing/interpolation enabled")
        print(f"🤖 Gripper processing: {'Hysteresis' if self.use_hysteresis else 'Threshold'} mode")

        try:
            metadata = self.policy.get_server_metadata()
            print(f"📋 Server metadata: {metadata}")
        except Exception as e:
            print(f"⚠️  Could not get server metadata: {e}")

    # ---------- Gripper action processing ----------
    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Action의 gripper 부분을 binary로 처리"""
        if self.use_hysteresis:
            return self._process_gripper_action_hysteresis(action)
        else:
            return self._process_gripper_action_threshold(action)
    
    def _process_gripper_action_threshold(self, action: np.ndarray) -> np.ndarray:
        """Threshold 기반 gripper 처리"""
        processed_action = action.copy()
        gripper_idx = -1
        
        if action[gripper_idx] > self.gripper_threshold:
            processed_action[gripper_idx] = 1.0  # Close
        else:
            processed_action[gripper_idx] = 0.0  # Open
        
        return processed_action
    
    def _process_gripper_action_hysteresis(self, action: np.ndarray) -> np.ndarray:
        """Hysteresis 기반 gripper 처리 (노이즈 방지)"""
        processed_action = action.copy()
        gripper_idx = -1
        
        if action[gripper_idx] > 0.7:  # Close threshold
            processed_action[gripper_idx] = 1.0
            self._current_gripper_state = 1.0
        elif action[gripper_idx] < 0.3:  # Open threshold
            processed_action[gripper_idx] = 0.0
            self._current_gripper_state = 0.0
        else:
            # Hysteresis zone: 현재 상태 유지
            processed_action[gripper_idx] = self._current_gripper_state
        
        return processed_action

    # ---------- RTC helpers ----------
    def _soft_blend_overlap(self, prev_tail: np.ndarray, new_head: np.ndarray) -> np.ndarray:
        O = prev_tail.shape[0]
        if O <= 0:
            return new_head
        idx = np.arange(O, dtype=np.float32)
        alpha_new = 1.0 - np.exp(-(idx / max(1.0, O - 1)) / max(1e-6, self._overlap_tau))
        alpha_new = alpha_new.reshape(-1, 1)
        alpha_old = 1.0 - alpha_new
        return alpha_old * prev_tail + alpha_new * new_head

    def _start_background_chunk_generation(self, obs: dict):
        if self.is_generating_next_chunk:
            return
        self.is_generating_next_chunk = True

        def generate_next_chunk():
            try:
                frozen_actions = self._prepare_frozen_actions_for_inpainting()
                formatted_obs = self._format_observation_for_policy(obs)
                if hasattr(self.policy, "infer_with_flow_inpainting"):
                    result = self.policy.infer_with_flow_inpainting(
                        obs=formatted_obs,
                        frozen_actions=frozen_actions,
                        inpaint_horizon=self.open_loop_horizon,
                    )
                else:
                    result = self.policy.infer(formatted_obs)
                if "actions" in result:
                    with self._lock:
                        self.next_chunk_buffer = np.asarray(result["actions"], dtype=np.float32)
                    print(f"  ✅ Background: next chunk ready ({len(self.next_chunk_buffer)} actions)")
                else:
                    print("  ❌ Background: no actions in response")
            except Exception as e:
                print(f"  ❌ Background generation failed: {e}")
            finally:
                self.is_generating_next_chunk = False

        self.chunk_generation_thread = threading.Thread(target=generate_next_chunk, daemon=True)
        self.chunk_generation_thread.start()

    def _switch_to_next_chunk(self):
        with self._lock:
            if self.next_chunk_buffer is not None:
                print("  🔄 RTC: switching to pre-generated chunk")
                self.action_buffer = self.next_chunk_buffer
                self._prev_chunk = self.action_buffer.copy()
                self.action_idx = 0
                self.next_chunk_buffer = None
                self.interpolation_counter = 0
                if self.action_buffer is not None and len(self.action_buffer) > 0:
                    if self.current_action is None:
                        self.current_action = np.zeros_like(self.action_buffer[0], dtype=np.float32)
                    self.next_action = np.array(self.action_buffer[0], dtype=np.float32)
                return True
        return False

    # ---------- Agent core ----------
    def act(self, obs: dict) -> np.ndarray:
        try:
            current_joints = obs["joint_positions"]

            # 항상 진행률 기반 prefetch 검사
            if self.action_buffer is not None and len(self.action_buffer) > 0:
                progress = self.action_idx / float(len(self.action_buffer))
                if (not self.is_generating_next_chunk) and (progress >= self.rtc_prefetch_ratio):
                    self._start_background_chunk_generation(obs)

            # 새 청크 필요 시
            if self.action_buffer is None or self.action_idx >= len(self.action_buffer):
                print(f"\n🔄 RTC: requesting new actions (horizon={self.open_loop_horizon})")
                if not self._switch_to_next_chunk():
                    self._request_new_actions(obs)
                    if self.action_buffer is None:
                        return np.array(current_joints, dtype=np.float32)

                # 새 버퍼 시작 시 이전 청크와 soft-blend
                with self._lock:
                    if self._prev_chunk is not None and self.action_buffer is not None:
                        H = len(self.action_buffer)
                        E = min(self._emin, H)
                        O = max(0, H - E)
                        if O > 0 and len(self._prev_chunk) >= O:
                            prev_tail = np.asarray(self._prev_chunk[-O:], dtype=np.float32)
                            new_head = np.asarray(self.action_buffer[:O], dtype=np.float32)
                            blended = self._soft_blend_overlap(prev_tail, new_head)
                            self.action_buffer[:O] = blended

                # 인터폴레이션 초기화
                self.interpolation_counter = 0
                self.current_action = current_joints.copy()
                self.next_action = np.array(self.action_buffer[0], dtype=np.float32)

            # Interpolation
            if self.interpolation_counter < self.interpolation_steps:
                alpha_step = self.interpolation_counter
                interpolated_action = self._interpolate_action(self.current_action, self.next_action, alpha_step, self.interpolation_steps)
                self.interpolation_counter += 1
                smoothed_action = self._smooth_action(interpolated_action, current_joints)
                final_action = self._limit_velocity(smoothed_action, current_joints)
                # Gripper action processing 적용
                final_action = self._process_action(final_action)
                return final_action
            else:
                # 다음 액션으로
                self.action_idx += 1
                self.interpolation_counter = 0
                if self.action_idx < len(self.action_buffer):
                    self.current_action = self.next_action.copy()
                    self.next_action = np.array(self.action_buffer[self.action_idx], dtype=np.float32)
                    interpolated_action = self._interpolate_action(self.current_action, self.next_action, 0, self.interpolation_steps)
                    self.interpolation_counter = 1
                    smoothed_action = self._smooth_action(interpolated_action, current_joints)
                    final_action = self._limit_velocity(smoothed_action, current_joints)
                    # Gripper action processing 적용
                    final_action = self._process_action(final_action)
                    return final_action
                else:
                    # 버퍼 끝: pre-generated chunk로 전환 시도
                    if self._switch_to_next_chunk():
                        return self.act(obs)
                    return np.array(current_joints, dtype=np.float32)

        except Exception as e:
            print(f"❌ Policy inference failed: {e}")
            return np.array(obs["joint_positions"], dtype=np.float32)

    # ---------- motion helpers ----------
    def _interpolate_action(self, start_action: np.ndarray, end_action: np.ndarray, step: int, total_steps: int) -> np.ndarray:
        if total_steps <= 1:
            return end_action
        t = step / (total_steps - 1)
        eased = t * t * (3.0 - 2.0 * t)  # smoothstep
        return start_action + eased * (end_action - start_action)

    def _smooth_action(self, target_action: np.ndarray, current_joints: np.ndarray) -> np.ndarray:
        if self.last_smoothed_action is None:
            self.last_smoothed_action = current_joints.copy()
        a = float(self.smoothing_factor)
        smoothed = a * target_action + (1.0 - a) * self.last_smoothed_action
        self.last_smoothed_action = smoothed.copy()
        return smoothed

    def _limit_velocity(self, target_action: np.ndarray, current_joints: np.ndarray) -> np.ndarray:
        delta = target_action - current_joints
        max_v = float(self.max_velocity)
        delta = np.clip(delta, -max_v, max_v)  # per-joint clamp
        return current_joints + delta

    # ---------- policy I/O ----------
    def _request_new_actions(self, obs: dict):
        try:
            formatted_obs = self._format_observation_for_policy(obs)
            result = self.policy.infer(formatted_obs)
            if "actions" in result:
                actions = np.asarray(result["actions"], dtype=np.float32)
                # Gripper action processing 적용
                processed_actions = np.array([self._process_action(action) for action in actions])
                self.action_buffer = processed_actions
                self.action_idx = 0
                self._prev_chunk = processed_actions.copy()
                # 인터폴레이션 초기화는 caller에서 수행
            else:
                print("    ❌ no actions in response")
        except Exception as e:
            print(f"  ❌ failed to request new actions: {e}")
            import traceback
            traceback.print_exc()

    def _format_observation_for_policy(self, obs: dict) -> dict:
        base_image = obs.get("base_rgb")
        wrist_image = obs.get("wrist_rgb")

        def _to_rgb_or_zero(img, W=224, H=224):
            if img is None:
                return np.zeros((H, W, 3), dtype=np.uint8)
            if img.ndim == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = np.zeros((H, W, 3), dtype=np.uint8)
            if (img_rgb.shape[0], img_rgb.shape[1]) != (H, W):
                img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            return img_rgb

        base_image = _to_rgb_or_zero(base_image, 224, 224)
        wrist_image = _to_rgb_or_zero(wrist_image, 224, 224)

        formatted_obs = {
            "observation/base_image": base_image,
            "observation/wrist_image": wrist_image,
            "observation/state": np.asarray(obs["joint_positions"], dtype=np.float32),
            "prompt": self.task_prompt,
        }
        return formatted_obs

    def _prepare_frozen_actions_for_inpainting(self) -> np.ndarray:
        if self.action_buffer is None or self.action_idx <= 0:
            return np.array([])
        frozen_len = max(self.frozen_action_window, 0)
        start = max(0, self.action_idx - frozen_len)
        return np.asarray(self.action_buffer[start:self.action_idx], dtype=np.float32)


def main(args):
    if not GELLO_AVAILABLE:
        print("❌ Gello libraries are required")
        return
    if args.agent == "policy" and not OPENPI_AVAILABLE:
        print("❌ OpenPI client is required for policy agent")
        return

    # Cameras/robot init
    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) < 2:
            print("❌ Need at least 2 RealSense devices.")
            return
        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devs[:2]]
        drivers = [RealSenseDriver(s, args.camera_width, args.camera_height, args.camera_fps) for s in serials]

        ports = [args.wrist_camera_port, args.base_camera_port]
        for port, drv in zip(ports, drivers):
            threading.Thread(target=start_server, args=(port, drv), daemon=True).start()
        time.sleep(1)

        client1 = ZMQClientCamera(port=args.wrist_camera_port)
        client2 = ZMQClientCamera(port=args.base_camera_port)
        camera_clients = {"wrist": AsyncCamera(client1, crop_center=(320, 240)), "base": AsyncCamera(client2)}

        print("\n📷 DEBUG: Camera Status")
        time.sleep(2)
        for cam_name, cam in camera_clients.items():
            frame = cam.read()
            if frame is not None:
                print(f"  ✅ {cam_name} camera: {frame.shape}, {frame.dtype}")
            else:
                print(f"  ❌ {cam_name} camera: no frame")

        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)

    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)

    if args.agent == "policy":
        print("\n🎯 POLICY AGENT INITIALIZATION:")
        print(f"  📝 Task prompt: '{args.task_prompt}'")
        if not args.task_prompt or len(args.task_prompt.strip()) == 0:
            print("  ❌ ERROR: empty task prompt")
            return

        agent = PolicyAgent(args.policy_host, args.policy_port, args.api_key, args.task_prompt, args.open_loop_horizon)
        
        # PolicyAgent는 이미 하드코딩된 값들을 사용하므로 추가 설정 불필요
        # - smoothing_factor: 0.3 (하드코딩)
        # - max_velocity: 0.1 (하드코딩)  
        # - interpolation_steps: 3 (하드코딩)
        # - RTC 및 Flow inpainting: 자동 활성화

    elif args.agent == "gello":
        gello_port = args.gello_port
        if gello_port is None:
            usb_ports = glob.glob("/dev/serial/by-id/*")
            print(f"Found {len(usb_ports)} ports")
            if len(usb_ports) > 0:
                gello_port = usb_ports[0]
                print(f"using port {gello_port}")
            else:
                raise ValueError("No gello port found")
        if args.start_joints is None:
            reset_joints = np.deg2rad([0, -90, -90, -90, 90, 180])
        else:
            reset_joints = args.start_joints
        agent = GelloAgent(port=gello_port, start_joints=args.start_joints)

        curr_joints = env.get_obs()["joint_positions"]
        if reset_joints.shape == curr_joints.shape:
            max_delta = (np.abs(curr_joints - reset_joints)).max()
            steps = min(int(max_delta / 0.01), 100)
            for jnt in np.linspace(curr_joints, reset_joints, steps):
                env.step(jnt)
                time.sleep(0.001)

    elif args.agent in ("dummy", "none"):
        agent = DummyAgent(num_dofs=robot_client.num_dofs())
    else:
        raise ValueError(f"Invalid agent name: {args.agent}")

    # Start pose/soft start
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
        for i, delta, joint, current_j in zip(ids, abs_deltas[id_mask], start_pos[id_mask], joints[id_mask]):
            print(f"joint[{i}]: delta={delta:4.3f}, leader={joint:4.3f}, follower={current_j:4.3f}")
        return

    print(f"Start pos: {len(start_pos)} Joints: {len(joints)}")
    assert len(start_pos) == len(joints), f"agent output dim = {len(start_pos)}, env dim = {len(joints)}"

    max_delta = 0.05
    for _ in range(25):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        m = np.abs(delta).max()
        if m > max_delta:
            delta = delta / m * max_delta
        env.step(current_joints + delta)

    obs = env.get_obs()
    joints = obs["joint_positions"]
    action = agent.act(obs)
    if (action - joints > 0.5).any():
        print("Action is too big")
        joint_index = np.where(action - joints > 0.8)
        for j in joint_index:
            print(f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}")
        exit()

    if args.use_save_interface:
        from gello.data_utils.keyboard_interface import KBReset
        kb_interface = KBReset()

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    save_path = None
    buffer: List[Tuple[Dict[str, Any], np.ndarray]] = []
    recording: bool = False
    start_time = time.time()
    last_timing_log = 0.0

    while True:
        t_pass = time.time() - start_time
        print_color(f"\rTime passed: {round(t_pass, 2)}          ", color="white", attrs=("bold",), end="", flush=True)

        action = agent.act(obs)

        # 주기 디버그(10초마다)
        if int(t_pass) % 10 == 0:
            print(f"\n🔍 DEBUG(t={int(t_pass)}s) joints: {obs.get('joint_positions')[:3] if 'joint_positions' in obs else 'NA'}")

        if args.use_save_interface:
            state = kb_interface.update()
            if state == "start":
                dt_time = datetime.datetime.now()
                save_path = Path(args.data_dir).expanduser() / args.agent / dt_time.strftime("%m%d_%H%M%S")
                save_path.mkdir(parents=True, exist_ok=True)
                print(f"Recording to {save_path}")
                buffer.clear()
                recording = True
            elif state == "save":
                assert save_path is not None, "something went wrong"
                buffer.append((obs.copy(), action.copy()))
            elif state == "normal":
                if recording and save_path and buffer:
                    h5_file = save_path / "data.hdf5"
                    with h5py.File(h5_file, "w") as f:
                        grp = f.create_group("data")
                        keys = list(buffer[0][0].keys())
                        for key in keys:
                            data_arr = np.stack([b[0][key] for b in buffer], axis=0)
                            grp.create_dataset(key, data=data_arr, compression="gzip", compression_opts=4)
                        acts = np.stack([b[1] for b in buffer], axis=0)
                        grp.create_dataset("actions", data=acts, compression="gzip", compression_opts=4)

                    # mp4 저장
                    fps = 30
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    for cam in ("base", "wrist"):
                        frames = [b[0][f"{cam}_rgb"] for b in buffer]
                        h, w = frames[0].shape[:2]
                        video_path = save_path / f"{cam}_rgb.mp4"
                        vw = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                        for frame in frames:
                            vw.write(frame)
                        vw.release()

                    recording = False
                    buffer.clear()
                    save_path = None
            else:
                raise ValueError(f"Invalid state {state}")

        t0 = time.perf_counter()
        obs = env.step(action)
        dt_actual = time.perf_counter() - t0
        dt_target = 1.0 / args.hz
        sleep_time = dt_target - dt_actual
        if sleep_time > 0:
            time.sleep(sleep_time)
        total_time = time.perf_counter() - t0

        # 타이밍 로그는 2초에 1회
        if time.time() - last_timing_log > 2.0:
            print(f"[Timing] step={dt_actual*1000:.1f}ms, sleep={(max(sleep_time,0))*1000:.1f}ms, total={total_time*1000:.1f}ms (target={dt_target*1000:.1f}ms)")
            if sleep_time <= 0:
                print(f"[Warning] ⚠️ overrun: step {dt_actual*1000:.1f}ms > target {dt_target*1000:.1f}ms")
            last_timing_log = time.time()


if __name__ == "__main__":
    main(tyro.cli(Args))
