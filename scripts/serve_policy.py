import dataclasses
import enum
import logging
import socket
import threading

import tyro
import numpy as np
import jax
from jax import device_get

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    UR3 = "ur3"


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    pass


@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # RTC / Inpainting params
    enable_flow_inpainting: bool = True
    flow_inpaint_steps: int = 5
    min_exec_horizon: int = 25       # E_min
    soft_mask_tau: float = 0.25      # overlap blending 온도(0.1~0.5)
    seed: int = 0                    # RNG 카운터 시작값
    warmup: bool = True              # 첫 호출 JIT 워밍업


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(config="pi0_aloha", dir="gs://openpi-assets/checkpoints/pi0_base"),
    EnvMode.ALOHA_SIM: Checkpoint(config="pi0_aloha_sim", dir="gs://openpi-assets/checkpoints/pi0_aloha_sim"),
    EnvMode.DROID: Checkpoint(config="pi0_fast_droid", dir="gs://openpi-assets/checkpoints/pi0_fast_droid"),
    EnvMode.LIBERO: Checkpoint(config="pi0_fast_libero", dir="gs://openpi-assets/checkpoints/pi0_fast_libero"),
    EnvMode.UR3: Checkpoint(config="pi0_ur3_lora_finetune", dir="./checkpoints/pi0_ur3_lora_finetune/exp1/8999"),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            base_policy = _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            base_policy = create_default_policy(args.env, default_prompt=args.default_prompt)

    if args.enable_flow_inpainting and hasattr(base_policy, "model") and hasattr(base_policy.model, "sample_actions"):
        print(f"🚀 Flow-based inpainting enabled: steps={args.flow_inpaint_steps}, Emin={args.min_exec_horizon}")
        return FlowInpaintingPolicy(
            base_policy,
            flow_steps=args.flow_inpaint_steps,
            min_exec_horizon=args.min_exec_horizon,
            soft_mask_tau=args.soft_mask_tau,
            seed=args.seed,
            do_warmup=args.warmup,
        )
    return base_policy


class FlowInpaintingPolicy(_policy.Policy):
    """PI0 + RTC-like wrapper: freeze E_min, overlap soft-blend, safe JAX→NumPy, RNG counter, warmup."""

    def __init__(
        self,
        base_policy: _policy.Policy,
        flow_steps: int = 5,
        min_exec_horizon: int = 25,
        soft_mask_tau: float = 0.25,
        seed: int = 0,
        do_warmup: bool = True,
    ):
        super().__init__()
        self.base_policy = base_policy
        self.flow_steps = int(flow_steps)
        self.metadata = base_policy.metadata

        self.min_exec_horizon = int(min_exec_horizon)
        self.soft_mask_tau = float(soft_mask_tau)

        self._rng_counter = np.uint32(seed)
        self._prev_chunk = None  # np.ndarray [H, D]
        self._H = None
        self._D = None
        self._lock = threading.Lock()

        if do_warmup and hasattr(self.base_policy, "model") and hasattr(self.base_policy.model, "sample_actions"):
            try:
                dummy_obs = self._dummy_pi0_obs()
                _ = self.base_policy.model.sample_actions(
                    rng=jax.random.PRNGKey(int(self._rng_counter)),
                    observation=dummy_obs,
                    num_steps=max(1, self.flow_steps),
                )
                self._rng_counter += 1
                print("[warmup] JIT compile done.")
            except Exception as e:
                print(f"[warmup] skipped: {e}")

    def infer(self, observation: dict) -> dict:
        out = self.base_policy.infer(observation)
        acts = out.get("actions", None)
        if acts is None:
            return out

        acts = self._to_np(acts)
        acts = self._ensure_batch_and_squeeze(acts)  # (H,D)

        if self._H is None:
            self._H, self._D = acts.shape

        refined = None
        if hasattr(self.base_policy, "model") and hasattr(self.base_policy.model, "sample_actions"):
            try:
                pi0_obs = self._convert_to_pi0_format(observation)
                refined_jax = self.base_policy.model.sample_actions(
                    rng=jax.random.PRNGKey(int(self._rng_counter)),
                    observation=pi0_obs,
                    num_steps=max(1, self.flow_steps),
                )
                self._rng_counter += 1
                refined = self._ensure_batch_and_squeeze(self._to_np(refined_jax))
            except Exception as e:
                print(f"❌ flow-based sampling failed: {e}")

        if refined is not None and refined.shape == acts.shape:
            H = refined.shape[0]
            E = min(self.min_exec_horizon, H)
            O = max(0, H - E)

            current_chunk = refined.copy()
            # overlap soft blend with previous chunk
            if self._prev_chunk is not None and O > 0 and len(self._prev_chunk) >= O:
                prev_tail = self._prev_chunk[-O:, :]
                new_head = current_chunk[:O, :]
                blended = self._soft_blend_overlap(prev_tail, new_head)
                current_chunk[:O, :] = blended

            with self._lock:
                self._prev_chunk = current_chunk.copy()
            out["actions"] = current_chunk
            out["flow_inpainting_applied"] = True
            out["rtc"] = {"E_min": E, "overlap": O}
        else:
            with self._lock:
                self._prev_chunk = acts.copy()
            out["flow_inpainting_applied"] = False

        return out

    # ---------- helpers ----------
    def _to_np(self, arr):
        try:
            return np.asarray(device_get(arr))
        except Exception:
            return np.asarray(arr)

    def _ensure_batch_and_squeeze(self, arr: np.ndarray) -> np.ndarray:
        # allow (H,D) or (1,H,D)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"actions must be (H,D) or (1,H,D), got {arr.shape}")
        return arr.astype(np.float32, copy=False)

    def _soft_blend_overlap(self, prev_tail: np.ndarray, new_head: np.ndarray) -> np.ndarray:
        O = prev_tail.shape[0]
        if O <= 0:
            return new_head
        idx = np.arange(O, dtype=np.float32)
        alpha_new = 1.0 - np.exp(-(idx / max(1.0, O - 1)) / max(1e-6, self.soft_mask_tau))
        alpha_new = alpha_new.reshape(-1, 1)
        alpha_old = 1.0 - alpha_new
        return alpha_old * prev_tail + alpha_new * new_head

    def _convert_to_pi0_format(self, observation: dict):
        """Client observation을 PI0 모델 형식으로 변환"""
        try:
            # UR3 observation을 PI0 형식으로 변환
            pi0_obs = {
                "images": {
                    "base_0_rgb": observation.get("observation/base_image"),
                    "left_wrist_0_rgb": observation.get("observation/wrist_image"),
                    "right_wrist_0_rgb": np.zeros_like(observation.get("observation/wrist_image")),
                },
                "image_masks": {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    "right_wrist_0_rgb": np.False_
                },
                "state": observation.get("observation/state"),
                "prompt": observation.get("prompt"),
            }
            return pi0_obs
        except Exception as e:
            print(f"❌ Observation conversion failed: {e}")
            return observation

    def _dummy_pi0_obs(self):
        """UR3에 맞는 dummy observation"""
        return {
            "images": {
                "base_0_rgb": np.zeros((1, 224, 224, 3), np.float32),
                "left_wrist_0_rgb": np.zeros((1, 224, 224, 3), np.float32),
                "right_wrist_0_rgb": np.zeros((1, 224, 224, 3), np.float32),
            },
            "image_masks": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_
            },
            "state": np.zeros((1, 7), np.float32),  # UR3: 7 DOF
            "prompt": "pick up the green grape and put it in the gray pot",
        }


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
